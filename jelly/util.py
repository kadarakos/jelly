from enum import Enum


class EnumChoices:
    enumeration: Enum
    options: Enum

    def __init__(self, name: str, enumeration: Enum):
        self.enumeration = enumeration
        options = {x.name: x.name for x in enumeration}
        self.options = Enum(name, options)

    def resolve(self, option: str):
        return getattr(self.enumeration, option).value
