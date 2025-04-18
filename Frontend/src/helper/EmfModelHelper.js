import dslConfigurationDetails from './dslConfigurationDetails.js';
import EntityMetadataAssociator from '../emfMetadata/EntityMetadataAssociator';

class EmfModelHelper {

    /**
     * Flattens the hierarchical structure of the given EMF-Model (from backend).
     *
     * @param emfModelTree: {...} //from backend
     * @returns     [{
     *                  data: EmfModelData,
     *                  parent: ,
     *                  children: ,
     *                  metadata: look up EntityMetadataAssociator
     *                  self: Property-name in parent-emf-entity. Used to localize entity in emf-tree. 
     *                        Important for creating associations between entities in backend.
     *              },
     *              ...]
     */
    static flattenEmfModelTree(emfModelTree) {
        let flatEmfModel = [];

        this.recursion(flatEmfModel, emfModelTree, undefined, undefined);
        return flatEmfModel;
    }

   

    /**
     * Returns a list of entity-pairs wich are renderable and associated to each other. 
     * 
     * @param emfModelFlat: output from EmfModelHelper.getRenderableEmfEntities(...)
     * @returns [{
     *              source: ,
     *              target: 
     *          }, 
     *          ...]
     */
    static getEmfEntityAssociations(emfModelFlat) {
        let associationContainer = [];

        associationContainer = [...associationContainer, ...EmfModelHelper.getMethodToMethodAssociations(emfModelFlat)];
        associationContainer = [...associationContainer, ...EmfModelHelper.getAssignmentAssociations(emfModelFlat)];
        associationContainer = [...associationContainer, ...EmfModelHelper.getPlaceHolderToFunktionAssociations(emfModelFlat)];
        
        return associationContainer;
    }

    static getMethodToMethodAssociations(emfModelFlat) {
        let associationContainer = [];

        let renderableEntities = EmfModelHelper.getRenderableEmfEntities(emfModelFlat);
        for(let entity of renderableEntities) {
            let renderableParent = EmfModelHelper.getRenderableParent(entity);
            if(renderableParent) {
                associationContainer.push({source: entity, target: renderableParent});
            }
        }
        return associationContainer;
    }

    static getAssignmentAssociations(emfModelFlat) {
        let associationContainer = [];

        let renderableEntities = EmfModelHelper.getRenderableEmfEntities(emfModelFlat);
        for(let entity of renderableEntities) {
            if(entity.parent && entity.parent.data.className && entity.parent.data.className.endsWith('SmlAssignment')) {
                let assignees = entity.parent.getChild('@assigneeList').children;
                for(let assignee of assignees) {
                    associationContainer.push({source: entity, target: assignee});
                }
            }
        }
        return associationContainer;
    }

    static getPlaceHolderToFunktionAssociations(emfModelFlat) {
        let associationContainer = [];

        let referenceableEmfEntities = EmfModelHelper.getReferenceableEmfEntities(emfModelFlat);
        for(let entity of referenceableEmfEntities) {
            let referredEntity = EmfModelHelper.getRenderableParentFromReference(emfModelFlat, entity);
            if(referredEntity) {
                associationContainer.push({source: referredEntity, target: EmfModelHelper.getRenderableParent(entity)});
            }
        }
        return associationContainer;
    }

    /**
     * Returns a list of renderable Entities. 
     * 
     * @param emfModelFlat: output from EmfModelHelper.flattenEmfModelTree(...)
     * @returns subcollection fo emfModelFlat
     */
    static getRenderableEmfEntities(emfModelFlat) {
        let renderableEntities = [];
        
        for(let entity of emfModelFlat) {
            if(entity.metadata !== undefined) {
                renderableEntities.push(entity);
            }
        }
        return renderableEntities;
    }

    /**
     * Returns entities witch are references by them self (in emf-model)
     * 
     * @param emfModelFlat 
     */
    static getReferenceableEmfEntities(emfModelFlat) {
        let referenceableEntities = [];

        for(let entity of emfModelFlat) {
            if(entity.data['$ref'] !== undefined && 
                !entity.data['$ref'].startsWith('file:') &&
                !entity.data['$ref'].startsWith('jar:file:')) {
                referenceableEntities.push(entity)
            }
        }
        return referenceableEntities;
    }

    /**
     * Returns the next renderable parent/grandparent (recursive).
     * 
     * @param emfEntity 
     */
    static getRenderableParent(emfEntity) {
        if(emfEntity.parent === undefined) {
            return undefined;
        } else if(emfEntity.parent.metadata !== undefined) {
            return emfEntity.parent;
        } else {
            return EmfModelHelper.getRenderableParent(emfEntity.parent);
        }
    }

    /**
     * Searches in the flat emf-model and returns the found entity (if renderable) or next renderable parent/grandparent.
     * 
     * @param emfEntity 
     */
    static getRenderableParentFromReference(emfModelFlat, emfEntity) {
        const traverseChildren = (entity, searchPathPieces) => {
            if(searchPathPieces.length === 0) {
                return entity;
            }
            
            const currentPathPiece = searchPathPieces.shift();
            
            return traverseChildren(entity.children.filter((childEntity) => {
                return childEntity.self === currentPathPiece;
            })[0], searchPathPieces)
        }

        const pathPieces = emfEntity.data['$ref'].replace('//', '').split('/');
        // find root-entity
        const rootEntity = emfModelFlat.filter((entity) => {
            return entity.self === pathPieces[0];
        })[0];

        // find entity in question in children
        pathPieces.shift();
        const foundEntity = traverseChildren(rootEntity, pathPieces);
        // could be the case that the found entity is not renderable
        if(foundEntity.metadata !== undefined) {
            return foundEntity;
        } else {
            return EmfModelHelper.getRenderableParent(foundEntity);
        }
    }

    /**
     * Creates a string witch represents the complete hierarchy up to the root-node (separated by '/').
     * Example: Workflow(statement[3])/.../ProcessCall(arguments[0])
     *          Workflow: ClassName of parent
     *          statement[3]: PropertyName from parent and array-index if is one
     *
     * @param emfEntity
     * @param suffix
     * @returns string
     */
    static getFullHierarchy(emfEntity, suffix = '') {
        let associationTargetPath = '//' + EmfModelHelper.getFullHierarchy2(emfEntity, suffix);
        associationTargetPath = associationTargetPath.substring(0, associationTargetPath.length - 1);
        return associationTargetPath;
    }

    /**
     *
     * @param emfEntity
     * @param suffix
     * @returns string
     */
    static getFullHierarchy2(emfEntity, suffix = '') {
        if(emfEntity.parent !== undefined)
            return this.getFullHierarchy2(emfEntity.parent, emfEntity.self + '/' + suffix);
        return suffix;
    }

    /**
     * Creates a different representation of EmfEntity for easier processing.
     *
     * @param emfEntity
     * @returns {{data: Object, arrays: Array, objects: Array}}
     */
    static createMetaObject(emfEntity) {
        let data = {};
        let arrays = [];
        let objects = [];

        for(let propertyName in emfEntity) {
            let property = emfEntity[propertyName];

            if(Array.isArray(property)) {
                arrays.push({
                    name: propertyName,
                    data: property
                });
            } else if(typeof property === 'object') {
                objects.push({
                    name: propertyName,
                    data: property
                });
            } else {
                data[propertyName] = property;
            }
        }
        return { data, arrays, objects };
    }

    /**
     * Traverses the given EmfModel-Hierarchy and creates an Array of all elements. 
     * The attribute metadata (if not undefined) is a indication that this entity can be rendered.
     *
     * @param flatEmfEntityContainer
     * @param emfEntity
     * @param parent
     * @param self
     * @param currentId         : has to be object-reference (increments for every object not for hierarchy-level)
     * @returns {data: Object, parent: Object, children: Array, metadata: Object, self: string}
     */
    static recursion(flatEmfEntityContainer, emfEntity, parent, self, currentId = {id: 0}) {
        if(emfEntity === undefined || emfEntity === null)
            return undefined;

        currentId.id++;

        let { data, arrays, objects } = this.createMetaObject(emfEntity);
        let flatEntity = { id: currentId.id, data, parent, children: [], self, metadata: undefined, getValue: {}};
        let associatedMetadata = EntityMetadataAssociator.getMetadata(emfEntity);

        if(associatedMetadata !== undefined) {
            flatEntity.metadata = associatedMetadata;
        }

        arrays.forEach((array) => {
            array.data.forEach((element, index) => {
                flatEntity.children.push(this.recursion(flatEmfEntityContainer, element, flatEntity, `@${array.name}.${index}`, currentId));
            });
        });
        objects.forEach((object) => {
            flatEntity.children.push(this.recursion(flatEmfEntityContainer, object.data, flatEntity, `@${object.name}`, currentId));
        });
        flatEntity.children = flatEntity.children.filter((item) => {
            return !(item === undefined || item === null)
        })

        flatEntity.getChild = function(name) {
            return this.children.find((child) => {
                return child.self === name;
            });
        };

        flatEntity.raw = emfEntity;
        
        flatEmfEntityContainer.push(flatEntity);
        
        // TODO: ausbauen/löschen
        //-----------------
        let dslConfigs = dslConfigurationDetails.filter(entity => entity.className === flatEntity.data.className)[0];
        if(dslConfigs !== undefined){
            flatEntity.getValue = dslConfigs.getValue;
        }
        //-----------------
        
        return flatEntity;
    }

    static getProcessEmfPathFromFlatEntities(flatEntityList) {
        const collection = [];

        for(const entity of flatEntityList) {
            if(entity.data['$ref'] && (
                entity.data['$ref'].startsWith('file:') ||
                entity.data['$ref'].startsWith('jar:file:'))) {
                collection.push(entity.data['$ref']);
            }
        }
        return new Set(collection);
    }
}

export default {
    flattenEmfModelTree: (emfModelTree) => EmfModelHelper.flattenEmfModelTree(emfModelTree),
    getFullHierarchy: (emfEntity) => EmfModelHelper.getFullHierarchy(emfEntity),
    getRenderableEmfEntities: (emfModelFlat) => EmfModelHelper.getRenderableEmfEntities(emfModelFlat),
    getEmfEntityAssociations: (emfModelFlat) => EmfModelHelper.getEmfEntityAssociations(emfModelFlat),
    getProcessEmfPathFromFlatEntities: (emfModelFlat) => EmfModelHelper.getProcessEmfPathFromFlatEntities(emfModelFlat)
}
