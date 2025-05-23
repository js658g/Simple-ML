//node_modules
import { mxUtils, mxEvent, mxGraph, mxConnectionHandler, mxImage,  mxHierarchicalLayout, mxGraphView } from "mxgraph-js";
//helper
import EmfModelHelper from "../../../helper/EmfModelHelper";
import connectImage from "./../../../images/graph/association-arrow.png"
//services
import XtextServices from "../../../serverConnection/XtextServices";
import GraphServices from './mxGraphModelServices';

import reduxStore from './../../../reduxStore';
import { openContextMenu } from './../../../../reducers/contextMenu';
import configureStylesheet from "./mxgraphStylesheet"
import mxGraphConfig from "./mxGraphConfig";
import { openPropsEditor, closePropsEditor } from "../../../reducers/propsEditor";

class SMLGraph extends mxGraph {

    parent = this.getDefaultParent();
    config = {};
    EMFmodel = undefined;
    layout = {};

    constructor(parentContainer) {
        super(parentContainer);
        this.containerWidth = parentContainer.clientWidth;
        this.containerHeight = parentContainer.clientHeight;
        this.centerX = this.containerWidth/2;
        this.centerY = this.containerHeight/2;
        this.doResizeContainer(this.containerWidth-20, this.containerHeight-75)
        // disable default mxCellEditor
        if(this.cellEditor) {
            this.cellEditor.startEditing = () => {};
        }
    }

    /**
     *
     * @param {mxConstant} direction :  mxConstants.DIRECTION_NORTH or mxConstants.DIRECTION_WEST
     */
    initView(direction){
        this.layout = new mxHierarchicalLayout(this, direction);
        this.layout.parentBorder = 50;
        configureStylesheet(this);
        mxConnectionHandler.prototype.connectImage = new mxImage(connectImage,10,10);
        mxConnectionHandler.prototype.moveIconFront=true;
        this.layout.intraCellSpacing = 100;
        this.htmlLabels = true;
        this.setConnectable(true);
        this.labelDisplayOverride();
        this.addLabelperimeter();
        this.setCellsMovable(false);
        this.setCellsResizable(false);
    }

    /**
     * erases everything on graph.view
     */
    clear(){
        this.removeCells(this.getChildCells(this.parent, true, true));
    }

    /**
     * renders the model stored in this.EMFModel
     *
     */
    render(){ 
        if(this.EMFmodel === undefined)
            return;
        this.getModel().beginUpdate();
        try{
            var cells = this.addEntities(this.EMFmodel);
            cells.map(cell => {
                this.connectToParent(cell);
            });
            this.connectReferences(this.EMFmodel);
            this.layout.execute(this.parent);
            //TO-DO: Graph in der Mitte von Golden-layout.container platzieren und nicht in der MItte von der ganzen App
            //this.view.setTranslate(Math.floor(this.view.translate.x - this.getGraphBounds().x * this.view.scale + (this.containerWidth-this.getGraphBounds().width) * 0.3 / this.view.scale));
            //this.center(true, false, .3, 0.5)
        }
        finally {
            this.getModel().endUpdate();
        }
    }

    /**
     *
     * @param {JSON} flatModel EMFModel from DSL after flattening
     */
    updateEMFModel(flatModel){
        this.EMFmodel=flatModel;
    }

    /**
     *
     * @param {JSON} cellValue: Value from EMFModel
     * @param {string} cellStyle: style from configs
     */
    addEntity(cellValue, cellStyle){
        return this.insertVertex(this.parent, null, cellValue, 10, 10, 47, 47, cellStyle);
    }

    /**
     * draws all entities stored in this.EMFModel
     *
     *
     * @param {*} model
     */
    addEntities(model){
        if( model === undefined )
            return [];
        var cells = [];
        model.forEach(entity=>{
            var encodedEntityValue = GraphServices.encode(entity);
            entity['visible'] = mxGraphConfig.isVisibleEntity(entity);
            var entityStyle = mxGraphConfig.getStyle(entity);
            if (entity['visible'] === true) {
                entity['cellObject'] = this.addEntity(encodedEntityValue, entityStyle);
                entity['cellObject'].setValue(entity);
                cells.push(entity['cellObject']);
            }
        });
        return cells;
    }

    /**
     *
     * @param {mxCell} sourceCell
     * @param {mxCell} targetCell
     */
    addAssociation(sourceCell, targetCell){
        return this.insertEdge(this.parent, null, null, sourceCell, targetCell);
    }

    /**
     *
     * @param {JSON} model: EMFModel
     *  draws all '$ref' associations from this.EMFModel
     */
    connectReferences(model) {
        model.forEach(entity => {
            if (entity.data['$ref']){
                let target = GraphServices.findVisibleTargetCellInModel(entity);
                let decodedReference = GraphServices.decodeReference(entity.data);
                let source = GraphServices.findVisibleSourceCellInModel(decodedReference, this.EMFmodel);
                this.addAssociation(source, target);
            }
        })
    };

    returnAllEdges(){
        var edges = []
        for (var index in this.model.cells){
            if (this.model.cells[index].edge === true){
                edges.push(this.model.cells[index])
            }
        }
        return edges
    }
    returnAllVertices(){
        var vertices = []
        for (var index in this.model.cells){
            if (this.model.cells[index].vertex === true){
                vertices.push(this.model.cells[index])
            }
        }
        return vertices
    }

    returnAllSourceNodes(){
        var sourceNodes = [];
        var edges = this.returnAllEdges();
        edges.forEach(edge=>{
            sourceNodes.push(edge.source)
        })
        return sourceNodes;
    }

    isSourceNode(cell){
        var result = false;
        var sourceNodes = this.returnAllSourceNodes();
        var comparedSources=this.model.filterCells(sourceNodes, function(source){
            return source.id === cell.id
        })
        if (comparedSources.length !== 0){
            result = true;
        }
        return result;
    }

    /**
     * draws a connection between an mxCell and the next visible ancestor
     * @param {mxCell} cell with EMFEntity in cell.value
     * @returns void
     */
    connectToParent(cell){
        var parentCell = GraphServices.findVisibleParent(cell.value);
        if (parentCell !== undefined){
            this.addAssociation(cell, parentCell);
        }
    }

    removeDanglingEdges(){
        var cells = this.model.cells
        var edges = []
        for (var key in cells){
            if(cells[key].edge === true){
                edges.push(cells[key])
            }
        }
        edges.forEach(edge => {
            if (edge.source === null || edge.target === null ) {
                this.model.remove(edge)
            }
        })
    }


    addLabelperimeter(){
        // Redirects the perimeter to the label bounds if intersection
        // between edge and label is found
        var mxGraphViewGetPerimeterPoint = mxGraphView.prototype.getPerimeterPoint;
        mxGraphView.prototype.getPerimeterPoint = function(terminal, next, orthogonal, border)
        {
            var point = mxGraphViewGetPerimeterPoint.apply(this, arguments);
            
            if (point != null)
            {
                var perimeter = this.getPerimeterFunction(terminal);
                if (terminal.text != null && terminal.text.boundingBox != null)
                {
                    // Adds a small border to the label bounds
                    var b = terminal.text.boundingBox.clone();
                    b.grow(3)
                    if (mxUtils.rectangleIntersectsSegment(b, point, next))
                    {
                        point = perimeter(b, terminal, next, orthogonal);
                    }
                }
            }
            
            return point;
        };
    }

    /**
     *
     * @param {JSON} entityValue: EMFEntity
     * @returns {mxCell} with label "object" and attributes from EntityValue
     */
    encodeToMxCell(entityValue){
        const xmlDoc = mxUtils.createXmlDocument();
        var newMxCell = xmlDoc.createElement("object");
        for (let prop in entityValue) {
            newMxCell.setAttribute(prop, entityValue[prop]);
        }
        return newMxCell;
    }

    /**
     *
     */
    labelDisplayOverride(){
        this.convertValueToString=(cell) =>{
            if (cell.isVertex()) {
                return mxGraphConfig.getLabelName(cell);
            }
        }
    }


    addDeleteOnDoubleClickListener(){
        this.addListener(mxEvent.DOUBLE_CLICK, function(sender, evt){
            let cell = evt.getProperty('cell');
            reduxStore.dispatch(openContextMenu(cell, evt.properties.event.pageX, evt.properties.event.pageY))
        });
    }

    openPropsEditorOnClickListener(){
        this.addListener(mxEvent.CLICK, function(sender, evt){
            if (evt.getProperty('cell') !== undefined && evt.getProperty('cell').vertex === true){
                let cell = evt.getProperty('cell');
                reduxStore.dispatch(openPropsEditor(cell))
            } else 
            {
                reduxStore.dispatch(closePropsEditor())
            }
        });
    }

    addCreateAssociationListener(){
        this.connectionHandler.addListener(mxEvent.CONNECT, function (sender,evt, graph){
            var target = evt.properties.target;
            var source = evt.properties.cell.source;
            if (target !== undefined){
                /*reacts to adding of a new edge in existing view, validate via xtext and re-render the view */
                var edge = evt.getProperty('cell');
                var sourceEntity=edge.source.value;
                var targetEntity=edge.target.value;
                var from=EmfModelHelper.getFullHierarchy(sourceEntity);
                var to=EmfModelHelper.getFullHierarchy(targetEntity);
                XtextServices.createAssociation(from,to);
            } else {
                reduxStore.dispatch(openContextMenu(source, evt.properties.event.pageX, evt.properties.event.pageY))
            }

        })
    }


    addGraphListeners(){
        this.addCreateAssociationListener();
        this.addDeleteOnDoubleClickListener();
        this.openPropsEditorOnClickListener();
    };


}
export default SMLGraph
