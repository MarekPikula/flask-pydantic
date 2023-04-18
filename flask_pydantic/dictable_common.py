import json
from dataclasses import asdict
from typing import Any, ClassVar, Dict, Type, TypeVar, Union

from pydantic import BaseModel
from pydantic.json import pydantic_encoder
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class DataClassProtocol(Protocol):
    """Dataclass protocol used for type hinting."""

    __dataclass_fields__: ClassVar[Dict[str, Any]]


T = TypeVar("T")


@runtime_checkable
class DictableProtocol(Protocol):
    """Protocol for any class that implements from_dict() and to_dict() funtions."""

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create object from arbitrary dictrionary."""
        ...

    def to_dict(self, *args: Any, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dictionary from self."""
        ...


DictableModel = Union[BaseModel, DictableProtocol, DataClassProtocol]
DictableModelTuple = (BaseModel, DictableProtocol, DataClassProtocol)


def exclude_none_filter(source: Dict[str, Any]) -> Dict[str, Any]:
    """Exclude None recursively for generic dictionary."""
    return {
        key: exclude_none_filter(source) if isinstance(value, dict) else value
        for key, value in source.items()
        if value is not None
    }


def make_json_from_model(
    model: DictableModel, by_alias: bool, exclude_none: bool = False
) -> str:
    if isinstance(model, BaseModel):
        return model.json(exclude_none=exclude_none, by_alias=by_alias)

    if isinstance(model, DictableProtocol):
        out = model.to_dict()
    else:
        out = asdict(model)

    if exclude_none:
        out = exclude_none_filter(out)
    return json.dumps(out, default=pydantic_encoder)
