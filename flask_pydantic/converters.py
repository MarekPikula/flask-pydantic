from dataclasses import fields, is_dataclass
from typing import Dict, List, Optional, Type, Union

from pydantic import BaseModel
from pydantic.config import BaseConfig
from pydantic.fields import ModelField
from werkzeug.datastructures import MultiDict

from .dictable_common import DictableModel


def convert_query_params(
    query_params: "MultiDict[str, str]", model: Type[DictableModel]
) -> Dict[str, Union[str, List[str]]]:
    """
    group query parameters into lists if model defines them

    :param query_params: flasks request.args
    :param model: query parameter's model
    :return: resulting parameters
    """
    model_fields: Optional[Dict[str, ModelField]] = None
    if is_dataclass(model):
        model_fields = {
            field.name: ModelField(
                name=field.name,
                type_=field.type,
                class_validators=None,
                model_config=BaseConfig,
                default=field.default,
            )
            for field in fields(model)
        }
    elif issubclass(model, BaseModel):
        model_fields = model.__fields__

    return {
        **query_params.to_dict(),
        **(
            {
                key: value
                for key, value in query_params.to_dict(flat=False).items()
                if key in model_fields and model_fields[key].is_complex()
            }
            if model_fields is not None
            else {}
        ),
    }
