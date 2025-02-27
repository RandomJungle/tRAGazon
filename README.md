Experimenting with RAG on a [dataset from Kaggle](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)

```
📁 tRAGazón
|
└───📁 data_preparation: 
|   |   df_transform.py: dataframe transformations
|   |   dim_reduction.py: dimensionality reduction of embeddings using tSNE
|   |   get_embeddings.py: query Mistral AI embeddings for dataset
└───📁 RAG: 
|   |   rag.py: vector store and RAG querying
|   |   self_query_rag.py: vector store and RAG with self query
```

Examples of results from RAG :

> **What is a good recipes that uses strawberries, green tea and stilton cheese ?**
> 
> I'm sorry, based on the provided context, there is no recipe that combines strawberries, green tea, and Stilton cheese. The recipes given are for "Sherried Stilton and Green Peppercorn Spread", "Mixed Green Salad with Strawberry Dressing", and "Cheese Board with Roasted Strawberries, Garlic & Herb Nuts, and Honey & Orange Roasted Figs", as well as instructions for a cheese platter with "Stilton with Della Robbia Fruits and Nuts". None of these recipes include green tea or combine all three ingredients you mentioned.

Using on purpose a list of ingredients that are unlikely to appear together, the context list is observed to not match exactly the question

> **Can you give me a list of 3 recipes that are similar to Minestrone ?**
> 
> Based on the provided context, here are three recipes that have similar ingredients and/or methods to Minestrone:
> 
> 1. Recipe for Minestrone with Cannellini Beans (first set of ingredients): This recipe includes dried cannellini beans, onion, leek, carrot, garlic, zucchini, cabbage, rosemary, diced tomatoes, kidney beans, elbow macaroni, and parmesan cheese.
>
> 2. Recipe for Minestrone (second set of ingredients): This recipe includes cannellini beans, onion, bay leaf, prosciutto ends, extra-virgin olive oil, celery, carrot, onion, leek, garlic, celery stalks, carrots, potato, zucchini, green beans, plum tomatoes, Tuscan kale, savoy cabbage, vegetable stock, parmigiano-reggiano cheese, prosciutto ends, red pepper flakes, coarse salt, and black pepper.
> 
> 3. Recipe for Minestrone (fourth set of ingredients): This recipe includes cannellini or borlotti beans, olive oil, onion, carrots, garlic, celery, potatoes, diced tomatoes, tomato sauce, red wine, bay leaves, Italian herb seasoning, chickpeas, green peas, parsley, salt, and freshly ground pepper.
> 
> These recipes all feature various combinations of vegetables, beans, and herbs, which are common in Minestrone.

The issue here is that, since the question is talking explicitly about minestrone, the top retrieved context recipes are all Minestrone recipes, so the answer does not extend beyond Minestrone

> **Give me salmon recipes that do not use any herbs**
> 
> Based on the provided context, the following recipes do not include any herbs:
>
>1. Roasted Salmon With Green Herbs: You can simply omit the herbs from this recipe.
>
> 2. Super Herbed Sautéed Salmon with Creamy Leeks and Bacon: This recipe does not include any herbs in the salmon portion.

Here the different parts are somwhat working against one another, since the question include herbs, the word is probably weighing in on the embedding of the question, and so the retriever is retrieving mostly herb-based recipes based on the similarity with that vector, collecting as its top answer only recipes that heavily accent the herb part in their title.

When fixing k at 10, The list is larger, which means there are a few recipes that enter the context that are not using herbs

> **Give me salmon recipes that do not use any herbs**
> 
>Based on the provided context, there are two salmon recipes that do not use any herbs:
>
>1. Roasted Salmon With Green Herbs: Although the name includes "green herbs," the recipe itself does not require any herbs. The herbs are used as a topping for the salmon, but they can be omitted if desired. The actual recipe only includes salmon, salt, pepper, olive oil, lemon juice, and lemon wedges.
>
>2. Salmon Roasted in Butter: This recipe does not use any herbs. It includes salmon, butter, salt, pepper, and lemon wedges.

This could be a task for self querying, except that the more rigid logic of database querying means that the query will try to avoid the word 'herb' in the ingredient list, but won't descend or ascend the abstraction level, meaning the query will result in something like : 

```
{
    "query": "salmon",
    "filter": "not(contain(ingredient_tags, \"herbs\"))"
}
```

Which won't match any specific herb, like thyme or basil. The logic of query and natural language do not completely blend together in this specific case.