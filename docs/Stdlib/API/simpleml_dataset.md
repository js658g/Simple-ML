# Package `simpleml.dataset`

[Tutorial][tutorial] - [Idea and basic concepts][tutorial_concepts] | [Interface][tutorial_interface] | [**API**][api] | [DSL][dsl-tutorial]

[tutorial]: ../../Tutorial.md
[tutorial_concepts]: ../../Tutorial-Basic-Concepts.md
[tutorial_interface]: ../../Tutorial-The-Simple-ML-Interface.md
[api]: ./README.md
[dsl-tutorial]: ../../DSL/tutorial/README.md


## Table of Contents

* Classes
  * [`Dataset`](#class-Dataset)
  * [`Instance`](#class-Instance)
  * [`StandardNormalizer`](#class-StandardNormalizer)
  * [`StandardScaler`](#class-StandardScaler)
* Global functions
  * [`joinTwoDatasets`](#global-function-joinTwoDatasets)
  * [`loadDataset`](#global-function-loadDataset)
  * [`readDataSetFromCSV`](#global-function-readDataSetFromCSV)

----------

<a name='class-Dataset'/>

## Class `Dataset`
A dataset with its data instances (e.g., rows and columns).

**Constructor:** _Class has no constructor._

### `addAttribute` (Instance Method )
Add a new attribute to the dataset with values according to a transformation function

**Parameters:**
* `columnName: String` - _No description available._
* `transformFunc: (instance: Instance) -> value: Any` - _No description available._
* `newColumnLabel: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `addDayOfTheYearAttribute` (Instance Method )
Add a new attribute to the dataset specifying the day of the year of the specified column

**Parameters:**
* `columnName: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `addIsWeekendAttribute` (Instance Method )
Add a new attribute to the dataset specifying if the dates of the specified column are on the weekend or not

**Parameters:**
* `columnName: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `addWeekDayAttribute` (Instance Method )
Extract week day from given date attribute and add new attribute in dataset with weekday name

**Parameters:**
* `columnName: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `dropAttribute` (Instance Method )
Remove the provided attribute from the dataset

**Parameters:**
* `attribute: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `dropAttributes` (Instance Method )
Remove list of columns provided in argument from dataset

**Parameters:**
* `vararg attributes: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `dropMissingValues` (Instance Method )
Drops instances with missing values in the specified attribute

**Parameters:**
* `attribute: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `exportDataAsFile` (Instance Method )
Export any dataset to CSV file

**Parameters:**
* `filePath: String` - _No description available._

**Results:** _None returned._

### `filterInstances` (Instance Method )
Filter dataset based on any specific value from a column

**Parameters:**
* `filterFunc: (instance: Instance) -> shouldKeep: Boolean` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `getRow` (Instance Method )
Get a specific row of a dataset

**Parameters:**
* `rowNumber: Int` - _No description available._

**Results:**
* `instance: Instance` - _No description available._

### `keepAttribute` (Instance Method )
Create a subset of a dataset with only the provided column

**Parameters:**
* `attribute: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `keepAttributes` (Instance Method )
Create a subset of a dataset with only list of columns provided in argument

**Parameters:**
* `vararg attributes: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `sample` (Instance Method )
Create a sample of a dataset

**Parameters:**
* `nInstances: Int` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `setTargetAttribute` (Instance Method )
Set the specified attribute as prediction target

**Parameters:**
* `targetAttribute: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

### `splitIntoTrainAndTest` (Instance Method )
Split dataset in train and test datasets

**Parameters:**
* `trainRatio: Float` - _No description available._
* `randomState: Int? = null` - _No description available._

**Results:**
* `train: Dataset` - _No description available._
* `test: Dataset` - _No description available._

### `splitIntoTrainAndTestAndLabels` (Instance Method )
Splits dataset into four datasets: train/test and labels/features

**Parameters:**
* `trainRatio: Float` - _No description available._
* `randomState: Int? = null` - _No description available._

**Results:**
* `xTrain: Dataset` - _No description available._
* `xTest: Dataset` - _No description available._
* `yTrain: Dataset` - _No description available._
* `yTest: Dataset` - _No description available._

### `transformDatatypes` (Instance Method )
Convert all column values into numbers

**Parameters:** _None expected._

**Results:**
* `dataset: Dataset` - _No description available._

### `transformDateToTimestamp` (Instance Method )
Convert date column values into timestamps

**Parameters:**
* `columnName: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._


----------

<a name='class-Instance'/>

## Class `Instance`
Individual instance in dataset

**Constructor:** _Class has no constructor._

### `getValue` (Instance Method )
To get value of instance

**Parameters:**
* `attribute: String` - _No description available._

**Results:**
* `value: Any` - _No description available._


----------

<a name='class-StandardNormalizer'/>

## Class `StandardNormalizer`
Data normalizer

**Constructor parameters:** _None expected._

### `normalize` (Instance Method )
Normalize whole dataset

**Parameters:**
* `dataset: Dataset` - _No description available._

**Results:**
* `normalizedDataset: Dataset` - _No description available._


----------

<a name='class-StandardScaler'/>

## Class `StandardScaler`
Data scaler

**Constructor parameters:** _None expected._

### `scale` (Instance Method )
Scale whole dataset

**Parameters:**
* `dataset: Dataset` - _No description available._

**Results:**
* `scaledDataset: Dataset` - _No description available._


## Global Functions

<a name='global-function-joinTwoDatasets'/>

## Global Function `joinTwoDatasets`
Join two dataset and returns merged single dataset

**Parameters:**
* `firstData: Dataset` - _No description available._
* `secondData: Dataset` - _No description available._
* `joinColumnName1: String` - _No description available._
* `joinColumnName2: String` - _No description available._
* `firstSuffix: String` - _No description available._
* `secondSuffix: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

<a name='global-function-loadDataset'/>

## Global Function `loadDataset`
Load dataset

**Parameters:**
* `datasetID: String` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

<a name='global-function-readDataSetFromCSV'/>

## Global Function `readDataSetFromCSV`
Read dataset directly from CSV file

**Parameters:**
* `fileName: String` - _No description available._
* `datasetId: String` - _No description available._
* `separator: String` - _No description available._
* `hasHeader: String` - _No description available._
* `nullValue: String` - _No description available._
* `datasetName: String` - _No description available._
* `coordinateSystem: Int = 3857` - _No description available._

**Results:**
* `dataset: Dataset` - _No description available._

----------

**This file was created automatically. Do not change it manually!**