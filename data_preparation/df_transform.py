from functools import reduce
from typing import Optional, Dict, List

import numpy as np
import pandas as pd


def read_json_dataframe(
        file_path: str,
        orient: Optional[str] = 'table') -> pd.DataFrame:
    if file_path.endswith('.jsonl'):
        dataframe = pd.read_json(
            path_or_buf=file_path,
            lines=True
        )
    else:
        dataframe = pd.read_json(
            path_or_buf=file_path,
            orient=orient
        )
    return dataframe


def chunk_dataframe(
        dataframe: pd.DataFrame,
        num_chunks: int) -> List[pd.DataFrame]:
    """
    Split dataframe into n chunks (mainly for LLM querying)
    """
    if num_chunks <= 1:
        return [dataframe]
    chunks = np.array_split(dataframe, num_chunks)
    return chunks


def df_pipeline(dataframe, functions):
    """
    Pipeline function to chain functions on a dataframe
    """
    return reduce(lambda d, f: f(d), functions, dataframe)


def basic_pipeline(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms applied to dataframe additional (no column removal)
    """
    df_pipeline(
        dataframe=dataframe,
        functions=[
            add_line_break_description_column,
            add_line_break_ingredients_column,
            add_line_break_instructions_column,
            add_line_break_tags_column,
            extract_one_cuisine_tag,
            extract_one_type_tag
        ])
    return dataframe


def replace_punctuation_with_line_break(text: str) -> str:
    return text.replace(
        ',', ',<br>'
    ).replace(
        '.', '.<br>'
    ).replace(
        '!', '!<br>'
    ).replace(
        '?', '?<br>'
    )


def add_line_break_to_str(
        dataframe: pd.DataFrame,
        text_column: str) -> pd.DataFrame:
    dataframe[f'{text_column}_br'] = dataframe[text_column].apply(
        lambda x: replace_punctuation_with_line_break(
            x.replace('\n', '<br>')
        ) if isinstance(x, str) else ''
    )
    return dataframe


def add_line_break_to_list(
        dataframe: pd.DataFrame,
        list_column: str) -> pd.DataFrame:
    dataframe[f'{list_column}_br'] = dataframe[list_column].apply(
        lambda x: '<br>'.join(x) if isinstance(x, list) else ''
    )
    return dataframe


def add_line_break_to_dict(
        dataframe: pd.DataFrame,
        dict_column: str) -> pd.DataFrame:
    dataframe[f'{dict_column}_br'] = dataframe[dict_column].apply(
        lambda x: '<br>'.join(
            [f'{key}{value}' for key, value in x.items()]
        ) if isinstance(x, dict) else ''
    )
    return dataframe


def add_line_break_description_column(
        dataframe: pd.DataFrame) -> pd.DataFrame:
    return add_line_break_to_str(dataframe, 'description')


def add_line_break_ingredients_column(
        dataframe: pd.DataFrame) -> pd.DataFrame:
    return add_line_break_to_list(dataframe, 'ingredients')


def add_line_break_instructions_column(
        dataframe: pd.DataFrame) -> pd.DataFrame:
    return add_line_break_to_dict(dataframe, 'instructions')


def add_line_break_tags_column(
        dataframe: pd.DataFrame) -> pd.DataFrame:
    return add_line_break_to_dict(dataframe, 'tags')


def simplify_tag(tag: str) -> str:
    match tag.lower():
        case 'soup/stew':
            return 'Stew'
        case 'savory pie and tart':
            return 'Pie'
        case 'stock':
            return 'Soup'
        case 'parfait':
            return 'Dessert'
        case 'pasta & noodles':
            return 'Pasta'
        case 'rice bowl' | 'pilaf' | 'paella' | 'fried rice' | 'risotto':
            return 'Rice'
        case 'crepe':
            return 'Pancake'
        case 'sauce' | 'stuffing':
            return 'Condiment'
        case 'escabèche':
            return 'Marinade'
        case 'nachos' | 'cracker' | 'crudité':
            return 'Dip'
        case 'ribs' | 'meatloaf' | 'meatball':
            return 'Meat'
        case 'bruschetta' | 'jam' | 'toast':
            return 'Spreads'
        case 'dough':
            return 'Bread'
        case 'granola' | 'snack bar':
            return 'Snack'
        case 'taco' | 'enchilada' | 'quesadilla' | 'tostada':
            return 'Tortilla'
        case 'burrito':
            return 'Wrap'
        case 'frittata':
            return 'Egg'
        case 'waffle':
            return 'Pastries'
        case 'fried chicken':
            return 'Fritter'
        case 'veggie burger':
            return 'Burger'
        case _:
            return tag


def find_first_tag(
        tags: Dict[str, List[str]] | float,
        tag_key: str,
        expected_tags: Optional[list] = None) -> str | float:
    if not isinstance(tags, dict):
        return 'None'
    key_tags = tags.get(tag_key)
    if not key_tags:
        return 'None'
    key_tags = [simplify_tag(tag) for tag in key_tags]
    if expected_tags:
        for tag in key_tags:
            if tag in expected_tags:
                return tag
    return key_tags[0]


def extract_one_cuisine_tag(
        dataframe: pd.DataFrame,
        tag_column: Optional[str] = 'tags') -> pd.DataFrame:
    dataframe['cuisine_tag'] = dataframe[tag_column].apply(
        lambda x: find_first_tag(x, 'cuisine')
    )
    return dataframe


def extract_one_type_tag(
        dataframe: pd.DataFrame,
        tag_column: Optional[str] = 'tags') -> pd.DataFrame:
    dataframe['type_tag'] = dataframe[tag_column].apply(
        lambda x: find_first_tag(
            x, 'type',
            expected_tags=[
                'Pasta', 'Salad', 'Stew', 'Beverages', 'Alcohol', 'Sauce',
                'Wrap', 'Soup', 'Pastries', 'Egg', 'Pie', 'Dessert', 'Rice',
                'Porridge', 'Condiment', 'Dip', 'Meat', 'Spreads', 'Bread',
                'Snack', 'Marinade', 'Fritter', 'Tortilla'
            ]
        )
    )
    return dataframe
