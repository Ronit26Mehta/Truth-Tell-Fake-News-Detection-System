# filepath: backend/knowledge_graph_manager.py
from neo4j import GraphDatabase
from config import Config

class KnowledgeGraphManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(Config.DATABASE_URI, auth=(Config.DATABASE_USER, Config.DATABASE_PASSWORD))

    def close(self):
        self.driver.close()

    def add_claim(self, claim, fact):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_claim, claim, fact)

    @staticmethod
    def _create_and_return_claim(tx, claim, fact):
        query = (
            "MERGE (c:Claim {text: $claim}) "
            "MERGE (f:Fact {text: $fact}) "
            "MERGE (c)-[:VERIFIED_BY]->(f) "
            "RETURN c, f"
        )
        result = tx.run(query, claim=claim, fact=fact)
        return result.single()

    def find_related_facts(self, claim):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_and_return_facts, claim)
            return [record["f.text"] for record in result]

    @staticmethod
    def _find_and_return_facts(tx, claim):
        query = (
            "MATCH (c:Claim {text: $claim})-[:VERIFIED_BY]->(f:Fact) "
            "RETURN f.text"
        )
        result = tx.run(query, claim=claim)
        return result