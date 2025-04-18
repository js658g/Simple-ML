import XtextServices from "../serverConnection/XtextServices";
import EmfModelHelper from "../helper/EmfModelHelper";
import TextEditorWrapper from "../components/EditorView/TextEditor/TextEditorWrapper";
import { debugInterface } from "./exposeToBrowserConsole";

let afterReactInit = () => {
  TextEditorWrapper.setText(
    "package test\n" +
      "import simpleml.collections.*\n" +
      "import simpleml.dataset.*\n" +
      "import simpleml.model.regression." +
      "\n" +
      "workflow predictSpeed {\n" +
      "\n" +
      "    // Load and prepare data\n" +
      '    val sample = loadDataset("SpeedAverages").sample(nInstances = 1000);\n' +
      "    val features = sample.keepAttributes(\n" +
      "        2  /* Floating Car Data point: has time (hour) */,\n" +
      "        3  /* Floating Car Data point: has time (day of week) */, \n" +
      "        4  /* Floating Car Data point: has time (month of year) */,\n" +
      "        6  /* Floating Car Data point: vehicle type (label) */,\n" +
      "        12 /* Street: type (label) */\n" +
      "    );\n" +
      "    val target = sample.keepAttributes(\n" +
      "        7  /* Floating Car Data point: has speed */\n" +
      "    );\n" +
      "\n" +
      "    // Define the model\n" +
      "    val model = Lasso(regularizationStrength = 0);\n" +
      "    \n" +
      "    // Train the model\n" +
      "    val trained_model = model.fit(features, target);\n" +
      "\n" +
      "    // Predict something and print the result\n" +
      "    val predictionFeatures = listOf(\n" +
      "        listOf(23, 3, 8, 1, 2)\n" +
      "    );\n" +
      "    val predictedTargets = trained_model.predict(features = predictionFeatures);\n" +
      "}"
  );

  TextEditorWrapper.setText(
    `package example

import simpleml.dataset.loadDataset
import simpleml.model.classification.DecisionTreeClassifier

workflow winebasic {

    // load data
    val df = loadDataset("WhiteWineQualityBinary");

    // Splitting the data into test and training sets
    val df_train, val df_test = df.splitIntoTrainAndTest(trainRatio=0.75, randomState=1);

    // split df_train and df_test into features and target
    val X_train = df_train.dropAttributes("quality");
    val X_test = df_test.dropAttributes("quality");
    val y_train = df_train.keepAttributes("quality");

    // Create estimator
    val estimator = DecisionTreeClassifier();

    // Train estimator and print results
    val model = estimator.fit(X_train, y_train);

    // Predict something with the model
    val y_pred = model.predict(X_test);
}`
  );

  // TextEditorWrapper.setText(
  //     "package example\n" +
  //     "\n" +
  //     "workflow main {\n" +
  //     "    val message = hello();\n" +
  //     "}\n" +
  //     "\n" +
  //     "step hello() -> message: String {\n" +
  //     '    yield message = "Hello, world!";\n' +
  //     "}\n"
  // );

  XtextServices.addSuccessListener((serviceType, result) => {
    debugInterface.d.lsr = result;
    if (result.emfModel) {
      let emfModel = JSON.parse(result.emfModel);
      debugInterface.d.emf = { inSync: true, data: emfModel };
      debugInterface.d.emf_flat = EmfModelHelper.flattenEmfModelTree(emfModel);
      debugInterface.d.emf_renderable = EmfModelHelper.getRenderableEmfEntities(
        debugInterface.d.emf_flat
      );
      debugInterface.d.emf_associations =
        EmfModelHelper.getEmfEntityAssociations(debugInterface.d.emf_flat);
    } else {
      debugInterface.d.emf.inSync = false;
    }
  });

  XtextServices.addSuccessListener((serviceType, result) => {
    console.log({ serviceType, result });
  });

  // val test = loadDataset("WhiteWineQualityBinary");

  TextEditorWrapper.setText(
    `package example

        import simpleml.dataset.loadDataset
        
        workflow exampleworkflow {
                    
            // val test1 = loadDataset("HannoverEvents");
            // val test2 = loadDataset("WhiteWineQualityBinary");
            val test3 = loadDataset("WhiteWineQuality");
            // val test4 = loadDataset("RedWineQualityBinary");
            // val test5 = loadDataset("RedWineQuality");
            // val test6 = loadDataset("FloatingCarData");
            // val test7 = loadDataset("PublicHolidaysGermany");
            // val test8 = loadDataset("RossmannStores");
            // val test9 = loadDataset("SchoolHolidaysNiedersachsen");
            // val test10 = loadDataset("PostOffices");
            val test11 = loadDataset("TrafficTweets");
            val speedAverages = loadDataset("SpeedAverages");

        }
        
        `
  );
};

export default afterReactInit;
