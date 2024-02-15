import copy
from dataclasses import dataclass
from typing import Dict


def fact_from_dict(fact_dict: Dict):
    if fact_dict["fact_parent"] is None:
        return Fact(**fact_dict)
    else:
        fact_dict = copy.deepcopy(fact_dict)
        fact_parent_entry = fact_dict.pop("fact_parent")
        fact_parent = Fact(**fact_parent_entry)
        return Fact(**fact_dict, fact_parent=fact_parent)


@dataclass
class Fact:
    subject: str
    rel_lemma: str
    object: str
    rel_p_id: str
    query: str
    fact_paragraph: str = None
    fact_parent: "Fact" = None
    intermediate_paragraph: str = None

    def get_subject(self) -> str:
        return self.subject

    def get_relation_property_id(self) -> str:
        return self.rel_p_id

    def get_object(self) -> str:
        return self.object

    def get_relation(self) -> str:
        return self.rel_lemma

    def get_paragraph(self) -> str:
        return self.fact_paragraph

    def get_intermediate_paragraph(self) -> str:
        return self.intermediate_paragraph

    def get_parent(self) -> "Fact":
        return self.fact_parent

    def get_query(self) -> str:
        return self.query

    def as_tuple(self):
        return self.subject, self.rel_p_id, self.object

    def as_dict(self):
        output = copy.deepcopy(self.__dict__)
        if self.fact_parent is not None:
            output["fact_parent"] = output["fact_parent"].as_dict()
        return output

    def __eq__(self, o: "Fact") -> bool:
        return o.subject == self.subject and o.object == self.object and o.rel_p_id == self.rel_p_id
