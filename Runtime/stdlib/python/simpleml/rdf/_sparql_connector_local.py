from __future__ import annotations

import json
import os
from io import StringIO

from rdflib import Graph
from rdflib.plugins.sparql.results.jsonresults import JSONResultSerializer
from rdflib.util import guess_format

# from rdflib_hdt import HDTStore

print("Init data catalog.")
graph = Graph()

dirName = os.path.dirname(__file__)

# if global_config.use_hdt:
# TODO(lr): removed for now since rdflib_hdt is not available on Windows systems
# folderPath = os.path.join(dirName, "../../../data_catalog/" + "OLD_data_catalog.hdt")
# graph = Graph(store=HDTStore(folderPath))
# graph.parse("***REMOVED***Runtime/stdlib/python/simpleml/rdf/data_catalog.xml",
#            format='xml')
#    pass
# else:
# graph.parse("***REMOVED***Runtime/stdlib/python/simpleml/rdf/data_catalog.rdf", format='application/rdf+xml')
folders = ["datasets", "external_vocabularies", "schema"]  # "ml_catalog", "schema"]

for folder in folders:
    baseFolder = os.getenv(
        "SML_DATA_CATALOG_PATH", os.path.join(dirName, "../../../data_catalog/")
    )
    folderPath = os.path.join(baseFolder, folder)

    for filename in os.listdir(folderPath):

        filename = os.path.join(folderPath, filename)

        format = "ttl"
        if folder != "datasets":
            format = guess_format(filename)

        graph.parse(filename, format=format)

# qres2 = graph.query("SELECT (COUNT(?a) AS ?cnt) WHERE { ?a ?b ?c }")
# for row in qres2:
#    print(row)

print("Init data catalog -> Done.")


def run_query(query_string):
    res = graph.query(query_string)
    json_stream = StringIO()
    JSONResultSerializer(res).serialize(json_stream)
    return json.loads(json_stream.getvalue())


def load_query(file_name, parameters=None, filter_parameters=None):
    if parameters is None:
        parameters = {}
    if filter_parameters is None:
        filter_parameters = []

    file_name_absolute = os.path.join(
        os.path.dirname(__file__), "../queries/" + file_name + ".sparql"
    )
    with open(file_name_absolute) as file:
        query = file.read()
    for key, value in parameters.items():
        query = query.replace("@" + key + "@", value)
    # Remove SPARQL comments
    for filter_parameter in filter_parameters:
        query = query.replace("#" + filter_parameter + " ", "")
    return query


def get_graph():
    return graph
