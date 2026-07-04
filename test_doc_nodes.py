from src.components.graph_builder import GraphBuilder

builder = GraphBuilder()
graph = builder.load()

print("GDPR EN" in graph.nodes())
print("Bosch Supplier Manual" in graph.nodes())
print("VW Annual Report 2023" in graph.nodes())