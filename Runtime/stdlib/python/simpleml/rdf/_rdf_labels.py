import simpleml.util.jsonLabels_util as config
from rdflib.namespace import XSD

lang = 'de'

statistics = 'statistics'
attributes = 'attributes'
topics = 'subjects'
title = 'title'
null_value = 'null_string'
separator = 'separator'
file_location = 'fileName'
has_header = 'hasHeader'
description = 'description'
id = 'id'
number_of_instances = 'number_of_instances'

attribute_label = 'label'

line_accurrence_in_areas = 'lineAccurrenceInAreas'
numberOfNullValues = 'numberOfNullValues'
decile = 'decile'
quartile = 'quartile'
averageNumberOfSpecialCharacters = 'averageNumberOfSpecialCharacters'
averageNumberOfTokens = 'averageNumberOfTokens'
numberOfValidValues = 'numberOfValidValues'
numberOfOutliersBelow = 'numberOfOutliersBelow'
averageNumberOfCapitalisedValues = 'averageNumberOfCapitalisedValues'
averageNumberOfDigits = 'averageNumberOfDigits'
numberOfOutliersAbove = 'numberOfOutliersAbove'
numberOfDistinctValues = 'numberOfDistinctValues'
histogram = 'histogram'
averageNumberOfCharacters = 'averageNumberOfCharacters'
median = 'median'
numberOfValues = 'numberOfValues'
mean = 'mean'
numberOfValidNonNullValues = 'numberOfValidNonNullValues'
maximum = 'maximum'
minimum = 'minimum'
standardDeviation = 'standardDeviation'
numberOfInvalidValues = 'numberOfInvalidValues'

value = 'value'
values = 'values'

valueDistribution = 'valueDistribution'
numberOfInstances = 'numberOfInstances'
bucketMinimum = 'bucketMinimum'
bucketMaximum = 'bucketMaximum'
bucketValue = 'value'

sample = 'sample_instances'
sample_header_labels = 'header_labels'
sample_lines = 'lines'

value_distribution = 'value_distribution'
value_distribution_number_of_instances = 'number_of_instances'
value_distribution_value = 'value'

xsd_data_types = { config.type_float: XSD.float, config.type_integer: XSD.integer, config.type_datetime: XSD.dateTime}