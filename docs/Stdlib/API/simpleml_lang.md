# Package `simpleml.lang`

[Tutorial][tutorial] - [Idea and basic concepts][tutorial_concepts] | [Interface][tutorial_interface] | [**API**][api] | [DSL][dsl-tutorial]

[tutorial]: ../../Tutorial.md
[tutorial_concepts]: ../../Tutorial-Basic-Concepts.md
[tutorial_interface]: ../../Tutorial-The-Simple-ML-Interface.md
[api]: ./README.md
[dsl-tutorial]: ../../DSL/tutorial/README.md


## Table of Contents

* Classes
  * [`Any`](#class-Any)
  * [`Boolean`](#class-Boolean)
  * [`Float`](#class-Float)
  * [`Int`](#class-Int)
  * [`Nothing`](#class-Nothing)
  * [`Number`](#class-Number)
  * [`String`](#class-String)
* Enums
  * [`AnnotationTarget`](#enum-AnnotationTarget)
* Annotations
  * [`Constant`](#annotation-Constant)
  * [`Deprecated`](#annotation-Deprecated)
  * [`Description`](#annotation-Description)
  * [`NoSideEffects`](#annotation-NoSideEffects)
  * [`Pure`](#annotation-Pure)
  * [`PythonModule`](#annotation-PythonModule)
  * [`PythonName`](#annotation-PythonName)
  * [`Repeatable`](#annotation-Repeatable)
  * [`Since`](#annotation-Since)
  * [`Target`](#annotation-Target)

----------

<a name='class-Any'/>

## Class `Any`
The common superclass of all classes.

**Constructor:** _Class has no constructor._


----------

<a name='class-Boolean'/>

## Class `Boolean`
A truth value.

**Constructor:** _Class has no constructor._


----------

<a name='class-Float'/>

## Class `Float`
A floating-point number.

**Constructor:** _Class has no constructor._


----------

<a name='class-Int'/>

## Class `Int`
An integer.

**Constructor:** _Class has no constructor._


----------

<a name='class-Nothing'/>

## Class `Nothing`
The common subclass of all classes.

**Constructor:** _Class has no constructor._


----------

<a name='class-Number'/>

## Class `Number`
A number.

**Constructor:** _Class has no constructor._


----------

<a name='class-String'/>

## Class `String`
Some text.

**Constructor:** _Class has no constructor._


<a name='enum-AnnotationTarget'/>

## Enum `AnnotationTarget`
The declaration types that can be targeted by annotations.
### Enum Variant `Annotation`
The annotation can be called on annotations.

**Parameters:** _None expected._


### Enum Variant `Attribute`
The annotation can be called on attributes.

**Parameters:** _None expected._


### Enum Variant `Class`
The annotation can be called on classes.

**Parameters:** _None expected._


### Enum Variant `CompilationUnit`
The annotation can be called on compilation units (i.e. files).

**Parameters:** _None expected._


### Enum Variant `Enum`
The annotation can be called on enums.

**Parameters:** _None expected._


### Enum Variant `EnumVariant`
The annotation can be called on enum variants.

**Parameters:** _None expected._


### Enum Variant `Function`
The annotation can be called on functions.

**Parameters:** _None expected._


### Enum Variant `Parameter`
The annotation can be called on parameters.

**Parameters:** _None expected._


### Enum Variant `Result`
The annotation can be called on results.

**Parameters:** _None expected._


### Enum Variant `Step`
The annotation can be called on steps.

**Parameters:** _None expected._


### Enum Variant `TypeParameter`
The annotation can be called on type parameters.

**Parameters:** _None expected._


### Enum Variant `Workflow`
The annotation can be called on workflows.

**Parameters:** _None expected._



<a name='annotation-Constant'/>

## Annotation `Constant`
Values assigned to this parameter must be constant.

**Valid targets:**
* Parameter

<a name='annotation-Deprecated'/>

## Annotation `Deprecated`
The declaration should no longer be used.

**Parameters:**
* `alternative: String? = null` - What to use instead.
* `reason: String? = null` - Why the declaration was deprecated.
* `sinceVersion: String? = null` - When the declaration was deprecated.
* `removalVersion: String? = null` - When the declaration will be removed.

**Valid targets:**
* Annotation
* Attribute
* Class
* Enum
* EnumVariant
* Function
* Parameter
* Result
* Step
* TypeParameter

<a name='annotation-Description'/>

## Annotation `Description`
The purpose of a declaration.

**Parameters:**
* `description: String` - The purpose of a declaration.

**Valid targets:**
* Annotation
* Attribute
* Class
* CompilationUnit
* Enum
* EnumVariant
* Function
* Parameter
* Result
* Step
* TypeParameter
* Workflow

<a name='annotation-NoSideEffects'/>

## Annotation `NoSideEffects`
The function has no side effects.

**Valid targets:**
* Function

<a name='annotation-Pure'/>

## Annotation `Pure`
The function has no side effects and returns the same results for the same arguments.

**Valid targets:**
* Function

<a name='annotation-PythonModule'/>

## Annotation `PythonModule`
The qualified name of the corresponding Python module (default is the qualified name of the package).

**Parameters:**
* `qualifiedName: String` - The qualified name of the corresponding Python module.

**Valid targets:**
* CompilationUnit

<a name='annotation-PythonName'/>

## Annotation `PythonName`
The name of the corresponding API element in Python (default is the name of the declaration in the stubs).

**Parameters:**
* `name: String` - The name of the corresponding API element in Python.

**Valid targets:**
* Attribute
* Class
* Enum
* EnumVariant
* Function
* Parameter
* Step
* Workflow

<a name='annotation-Repeatable'/>

## Annotation `Repeatable`
The annotation can be called multiple times for the same declaration.

**Valid targets:**
* Annotation

<a name='annotation-Since'/>

## Annotation `Since`
The version in which a declaration was added.

**Parameters:**
* `version: String` - The version in which a declaration was added.

**Valid targets:**
* Annotation
* Attribute
* Class
* CompilationUnit
* Enum
* EnumVariant
* Function
* Parameter
* Result
* Step
* TypeParameter
* Workflow

<a name='annotation-Target'/>

## Annotation `Target`
The annotation can target these declaration types. If the @Target annotation is not used any declaration type can be targeted.

**Parameters:**
* `vararg targets: AnnotationTarget` - The valid targets.

**Valid targets:**
* Annotation

----------

**This file was created automatically. Do not change it manually!**
