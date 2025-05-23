import XtextServices from "../serverConnection/XtextServices";
import EmfModelHelper from "../helper/EmfModelHelper";
import { showModal } from "../reducers/modal";
import DefaultModal from "../components/core/Modal/DefaultModal";
import store from "../reduxStore";

const debugInterface = {
  x: {
    //xtext
    s: {
      //services
      getEmfModel: () => XtextServices.getEmfModel(),
      getProcessMetadata: (entityPath) =>
        XtextServices.getProcessMetadata(entityPath),
      getProcessProposals: (entityId, entityPath) =>
        XtextServices.getProcessProposals(entityId, entityPath),
      createEntity: (entity) => XtextServices.createEntity(entity),
      deleteEntity: (entityPath) => XtextServices.deleteEntity(entityPath),
      createAssociation: (fromEntityPath, toEntityPath) =>
        XtextServices.createAssociation(fromEntityPath, toEntityPath),
      deleteAssociation: (fromEntityPath, toEntityPath) =>
        XtextServices.deleteAssociation(fromEntityPath, toEntityPath),
      getEntityAttributes: (entities) =>
        XtextServices.getEntityAttributes(entities),
      setEntityAttributes: (entity) =>
        XtextServices.setEntityAttributes(entity),
      generate: () => XtextServices.generate(),
      editProcessParameter: (index, value) => {
        const temp = {
          entityPath: EmfModelHelper.getFullHierarchy(
            store.getState().graphicalEditor.entitySelected
          ),
          parameterType: "string",
          parameterIndex: index,
          value: value,
        };
        XtextServices.editProcessParameter(temp);
      },
    },
    serviceObject: XtextServices,
  },
  h: {
    //helper
    flattenEmfModelTree: (emfModelTree) =>
      EmfModelHelper.flattenEmfModelTree(emfModelTree),
    getFullHierarchy: (emfEntity) => EmfModelHelper.getFullHierarchy(emfEntity),
    getFullHierarchy2: (emfEntity) =>
      EmfModelHelper.getFullHierarchy2(emfEntity),
    getSelectedEntity: () => {
      return store.getState().graphicalEditor.entitySelected;
    },
  },
  o: {
    //other
    showDefaultModal: () =>
      store.dispatch(
        showModal(DefaultModal, { text: "some text", message: "some message" })
      ),
  },
  d: {
    //data
    lsr: {}, //lastServiceResult
    emf: {},
    emf_flat: {},
    emf_renderable: {},
    l3s: {
      projectId: "",
      dataSets: {},
      dataSet: {},
    },
    e1: {
      //createEntity
      name: "test",
      className: "Assignment",
      value: "",
      children: [
        {
          name: "project",
          className: "ProcessCall",
          value: "",
          children: [
            {
              className: "StringLiteral",
              value: "someText",
            },
            {
              className: "IntegerLiteral",
              value: "23",
            },
          ],
        },
      ],
    },
    e2: {
      //createEntity
      name: "project",
      className: "ProcessCall",
      value: "",
      children: [
        {
          className: "StringLiteral",
          value: "someText",
        },
        {
          className: "IntegerLiteral",
          value: "23",
        },
      ],
    },
  },
};

let exposeToBrowserConsole = () => {
  window.deb = debugInterface;
};

export { debugInterface, exposeToBrowserConsole };
