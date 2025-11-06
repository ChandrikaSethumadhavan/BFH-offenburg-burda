# from transformers import MarianMTModel, MarianTokenizer
# import re
# import json
# from typing import List, Dict, Any, Iterable
# import re
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# import pandas as pd
# import numpy as np

# model_name = 'Helsinki-NLP/opus-mt-de-en'
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)
# def translate(text):
#     # Tokenize the input text
#     tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    
#     # Perform the translation
#     translation = model.generate(**tokenized_text)
    
#     # Decode the translated text
#     translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    
#     return translated_text
# def clean_spaces(s: str) -> str:
#     return re.sub(r"\s+", " ", str(s)).strip()

# def join_ingredient_parts(item: dict) -> str:
#     prefix = item.get("prefix") or ""
#     name = (item.get("ingredient") or {}).get("name", "") or ""
#     suffix = item.get("suffix") or ""
#     text = clean_spaces(f"{prefix} {name}")
#     if suffix:
#         text = clean_spaces(f"{text} {suffix}")
#     return text

# def extract_quantity_unit(item: dict) -> str:
#     q = item.get("quantity")
#     q_str = "" if q in (None, "") else str(q)
#     unit_obj = item.get("unit") or {}
#     unit = unit_obj.get("name") or unit_obj.get("short") or unit_obj.get("abbreviation") or ""
#     return clean_spaces(f"{q_str} {unit}")  # e.g. "200 g" or "" if none

# def join_quantity(item: dict) -> str:
#     return extract_quantity_unit(item)

# def iter_ingredients(recipe: dict) -> Iterable[dict]:
#     for block in recipe.get("ingredientBlocks", []):
#         for it in block.get("ingredients", []):
#             yield it
#     for section in recipe.get("content", []) or []:
#         for it in section.get("ingredients", []) or []:
#             yield it

# # --- load your JSON export (adjust filename as needed) ---
# with open("einfachkochen_export_800_recipes_1.json", "r", encoding="utf-8") as f:
#     data = json.load(f)
# recipes: List[Dict[str, Any]] = data.get("data", {}).get("recipeExport", [])

# # --- NEW: dict of {ingredient: quantity} for ONE recipe ---
# def ingredients_qty_dict_for_recipe(recipes: List[dict], recipe_name: str) -> Dict[str, str]:
#     """
#     Returns: { "Sugar": "200 g", "Milk": "100 ml", "Egg": "2" , ... }
#     If an ingredient appears multiple times, the last one wins.
#     """
#     recipe = next((r for r in recipes if r.get("name") == recipe_name), None)
#     if not recipe:
#         return {}
#     out: Dict[str, str] = {}
#     for it in iter_ingredients(recipe):
#         ing = join_ingredient_parts(it)
#         qty = join_quantity(it)  # already "number unit" or ""
#         out[ing] = qty
#     return out

# # --- OPTIONAL: dict of recipe -> {ingredient: quantity} for ALL recipes ---
# def ingredients_qty_dicts_all(recipes: List[dict]) -> Dict[str, Dict[str, str]]:
#     out: Dict[str, Dict[str, str]] = {}
#     for r in recipes:
#         rname = r.get("name", "Unnamed")
#         d: Dict[str, str] = {}
#         for it in iter_ingredients(r):
#             ing = join_ingredient_parts(it)
#             qty = join_quantity(it)
#             d[ing] = qty
#         out[rname] = d
#     return out

# # --- example usage ---
# one_recipe_map = ingredients_qty_dict_for_recipe(recipes, "Klassisches Gulasch")
# print(one_recipe_map)  # {'Mehl': '300 g', 'Zucker': '120 g', ...}
# all_recipes_map = ingredients_qty_dicts_all(recipes)
# # print(list(all_recipes_map.keys())[0:10])
# translated_texts_all = []
# original_texts_all = []
# def listing_original():
#     for t in one_recipe_map.keys():
#         original_texts_all.append(t)
#         return original_texts_all
# def translation_ing():
#     for src_texts in one_recipe_map.keys():
#         inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True)
#         translated = model.generate(**inputs)

#         # Decode and print result
#         translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
#         # print(translated_texts[0])
#         translated_texts_all.append(translated_texts[0])
#     return translated_texts
# def replacing_translation():
#     old_keys = list(one_recipe_map)  # preserves order of insertion
#     new_keys = translated_texts_all  # same length as old_keys

#     # 1) safest: build a new dict
#     renamed = {new: one_recipe_map[old] for old, new in zip(old_keys, new_keys)}
#     one_recipe_map = renamed
#     return one_recipe_map
# pck = 7  # adjust if you like
# TL_G = 5
# EL_G = 15
# def quantity_conversion(quantity: str):
#     s = quantity.strip().lower()
#     # grab the first number (works for "100 g", "100g", "1.5 kg", "250ml")
#     m = re.search(r'([\d.,]+)', s)
#     if not m:
#         return None
#     n = float(m.group(1).replace(',', '.'))  # <- n is just the number part

#     # convert to kg (simple assumptions)
#     if 'kg' in s:
#         convert = n
#     elif 'g' in s:
#         convert = n / 1000.0
#     elif 'ml' in s:
#         convert = n / 1000.0   # assume 1 ml ~ 1 g ~ 0.001 kg
#     elif 'l' in s:
#         convert = n 
#     elif " tl" in s or s.endswith("tl"):
#         convert = (n * TL_G) / 1000.0
#     elif " el" in s or s.endswith("el"):
#         convert = (n * EL_G) / 1000.0
#     elif " pck" in s:
#         convert = (n * pck) / 1000.0           # assume 1 L ~ 1 kg
#     else:
#         convert = n * 0.1         # unknown unit

#     return convert
# def normalize(word):
#     if isinstance(word,str):

#         word = word.lower()
#         word = word.split(",",1)[0]
        
#     else:
#         word = ""
#     return word

# def looking_up_co2():
    
#     df = pd.read_csv(r"C:\Users\muthu\Documents\bfh\BFH-offenburg-burda\final_co2_emissions_only_germany.csv",encoding="cp1252")
#     df = df.rename(columns={'Emissions (CO2 eq/kg)':'Emissions'})
#     df["norm"] = df["Product"].map(normalize)

#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     name_vecs = model.encode(df["norm"].tolist(), normalize_embeddings=True).astype("float32")
#     index = faiss.IndexFlatIP(name_vecs.shape[1])   # inner product == cosine for normalized vectors
#     index.add(name_vecs)
#     def embedding_lookup(name: str, addition, cos_cutoff=0.50):
#         q = model.encode([normalize(name)], normalize_embeddings=True).astype("float32")
#         D, I = index.search(q, 1)  # top-1
#         sim = float(D[0][0])
#         if sim < cos_cutoff:
#             return None, 0.0, addition  # no match → 0 CO₂

#         idx = int(I[0][0])
#         row = df.iloc[idx]
#         kg_co2 = float(row["Emissions"])

#         q_co2 = quantity_conversion(one_recipe_map[name])
#         if q_co2 is None:
#             return None, 0.0, addition  # missing quantity → 0 CO₂

#         total = kg_co2 * q_co2
#         addition += total

#         return row["norm"], total, addition
#     addition = 0
#     ingredients_list = []
#     co2_list = []
#     for i in list(one_recipe_map):
        
#         p,t,a = embedding_lookup(i,addition=addition)
#         addition = a
#         if p is None:
#             p = i
#             t = 0
#         # print(p,t,a)
#         ingredients_list.append(p)
#         co2_list.append(t)
#     return ingredients_list, co2_list
# def choosing_to_swap():
#     for i in range(len(co2_list)):
#     if (co2_list[i]/a)*100>15:
#         # print(i)
#         idx = i
        
#     ing_switch = ingredients_list[idx] 
#     co2_switch = co2_list[idx]
#     return ing_switch, co2_switch
# def getting_pro_carb():
#     df_macros_all = pd.read_csv(r"C:\Users\muthu\Documents\bfh\BFH-offenburg-burda\merged_semantic_ingredients.csv")
#     df_macros = pd.DataFrame().assign(Product = df_macros_all["Product"], protein = df_macros_all["protein"], carbs = df_macros_all["carbohydrate"],units= df_macros_all["mass_kg"],co2_per_kg=df_macros_all["Emissions (CO2 eq/kg)"])
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     df_macros = df_macros.reset_index(drop=True)
#     name_vecs = model.encode(df_macros["Product"].tolist(), normalize_embeddings=True).astype("float32")
#     index = faiss.IndexFlatIP(name_vecs.shape[1])   # inner product == cosine for normalized vectors
#     index.add(name_vecs)
#     def embedding_lookup(name: str, cos_cutoff=0.50):
#         q = model.encode([normalize(name)], normalize_embeddings=True).astype("float32")
#         D, I = index.search(q, 1)  # top-1
#         sim = float(D[0][0])
#         if sim < cos_cutoff:
#             return None, 0.0, addition  # no match → 0 CO₂

#         idx = int(I[0][0])
#         row = df_macros.iloc[idx]
#         protein = float(row["protein"])
#         carbs = float(row["carbs"])
#         unit = float(row["units"])

#         return row["Product"], row["carbs"] , row["protein"], row["units"]
#     product, carbs, protein, units = embedding_lookup(ing_switch)
#     mu = list(one_recipe_map)[idx]
#     quantity = one_recipe_map[mu]
#     s = quantity.strip().lower()
#     # grab the first number (works for "100 g", "100g", "1.5 kg", "250ml")
#     m = re.search(r'([\d.,]+)', s)

#     n = float(m.group(1).replace(',', '.'))
#     def convert(item, units):
#         item = ((n/1000)*units)*item
#         return item
#     carbs = convert(carbs,units)
#     protein = convert(protein, units)
#     return product, carbs,protein, units, n
# def rank_alternatives(target_name: str,
#                       df: pd.DataFrame,
#                       w_co2=0.4, w_protein=0.5, w_carbs=0.2,
#                       top_n=5):
#     """
#     Rank substitutes by combined CO₂ + nutritional similarity.
#     Columns needed: 'Product', 'co2_per_kg', 'protein', 'carbs'
#     """
#     # 1️⃣ Get target row
#     target = df[df["Product"].str.lower() == target_name.lower()]
#     if target.empty:
#         raise ValueError(f"{target_name} not found in dataset.")
#     target = target.iloc[0]
#     # print(target)
#     t_prot, t_carbs, t_co2 = target["protein"], target["carbs"], target["co2_per_kg"]

#     # 2️⃣ Normalize columns for fair comparison
#     df = df.copy()
#     df["norm_co2"] = df["co2_per_kg"] / df["co2_per_kg"].max()
#     df["protein_diff"] = abs(df["protein"] - t_prot) / df["protein"].max()
#     df["carbs_diff"]   = abs(df["carbs"] - t_carbs) / df["carbs"].max()

#     # 3️⃣ Weighted composite score (lower is better)
#     df["score"] = (w_co2 * df["norm_co2"] +
#                    w_protein * df["protein_diff"] +
#                    w_carbs * df["carbs_diff"])

#     # 4️⃣ Exclude target itself & only lower-CO₂ options
#     df = df[df["Product"].str.lower() != target_name.lower()]
#     df = df[df["co2_per_kg"] < t_co2]

#     # 5️⃣ Rank and return
#     best = df.sort_values("score").head(top_n)
#     return best[["Product", "co2_per_kg", "protein", "carbs", "score"]].reset_index(drop=True)
# df_best = rank_alternatives(ing_switch,df_macros)
# def replaced_lists():
#     ingredients_list[idx] = df_best["Product"][0]
#     co2_list[idx] = float((n/1000)*df_best["co2_per_kg"][0])
#     return ingredients_list, co2_list, sum(co2_list)


########################

# ==============================================
# Imports
# ==============================================
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
import re
import json
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Any, Iterable


# ==============================================
# Translation setup
# ==============================================
model_name = 'Helsinki-NLP/opus-mt-de-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate(text):
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return translated_text


# ==============================================
# Ingredient extraction helpers
# ==============================================
def clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def join_ingredient_parts(item: dict) -> str:
    prefix = item.get("prefix") or ""
    name = (item.get("ingredient") or {}).get("name", "") or ""
    suffix = item.get("suffix") or ""
    text = clean_spaces(f"{prefix} {name}")
    if suffix:
        text = clean_spaces(f"{text} {suffix}")
    return text


def extract_quantity_unit(item: dict) -> str:
    q = item.get("quantity")
    q_str = "" if q in (None, "") else str(q)
    unit_obj = item.get("unit") or {}
    unit = unit_obj.get("name") or unit_obj.get("short") or unit_obj.get("abbreviation") or ""
    return clean_spaces(f"{q_str} {unit}")


def join_quantity(item: dict) -> str:
    return extract_quantity_unit(item)


def iter_ingredients(recipe: dict) -> Iterable[dict]:
    for block in recipe.get("ingredientBlocks", []):
        for it in block.get("ingredients", []):
            yield it
    for section in recipe.get("content", []) or []:
        for it in section.get("ingredients", []) or []:
            yield it


# ==============================================
# Load recipe JSON and create ingredient maps
# ==============================================
with open("einfachkochen_export_800_recipes_1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

recipes: List[Dict[str, Any]] = data.get("data", {}).get("recipeExport", [])


def ingredients_qty_dict_for_recipe(recipes: List[dict], recipe_name: str) -> Dict[str, str]:
    recipe = next((r for r in recipes if r.get("name") == recipe_name), None)
    if not recipe:
        return {}
    out: Dict[str, str] = {}
    for it in iter_ingredients(recipe):
        ing = join_ingredient_parts(it)
        qty = join_quantity(it)
        out[ing] = qty
    return out


def ingredients_qty_dicts_all(recipes: List[dict]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for r in recipes:
        rname = r.get("name", "Unnamed")
        d: Dict[str, str] = {}
        for it in iter_ingredients(r):
            ing = join_ingredient_parts(it)
            qty = join_quantity(it)
            d[ing] = qty
        out[rname] = d
    return out


# ==============================================
# Translation setup for recipe
# ==============================================
one_recipe_map = ingredients_qty_dict_for_recipe(recipes, "Klassisches Gulasch")
translated_texts_all = []
original_texts_all = []


def listing_original():
    for t in one_recipe_map.keys():
        original_texts_all.append(t)
    return original_texts_all


def translation_ing():
    for src_texts in one_recipe_map.keys():
        inputs = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translated_texts_all.append(translated_texts[0])
    return translated_texts_all


def replacing_translation():
    old_keys = list(one_recipe_map)
    new_keys = translated_texts_all
    renamed = {new: one_recipe_map[old] for old, new in zip(old_keys, new_keys)}
    one_recipe_map.clear()
    one_recipe_map.update(renamed)
    return one_recipe_map


# ==============================================
# Quantity conversion
# ==============================================
pck = 7
TL_G = 5
EL_G = 15


def quantity_conversion(quantity: str):
    s = quantity.strip().lower()
    m = re.search(r'([\d.,]+)', s)
    if not m:
        return None
    n = float(m.group(1).replace(',', '.'))

    if 'kg' in s:
        convert = n
    elif 'g' in s:
        convert = n / 1000.0
    elif 'ml' in s:
        convert = n / 1000.0
    elif 'l' in s:
        convert = n
    elif " tl" in s or s.endswith("tl"):
        convert = (n * TL_G) / 1000.0
    elif " el" in s or s.endswith("el"):
        convert = (n * EL_G) / 1000.0
    elif " pck" in s:
        convert = (n * pck) / 1000.0
    else:
        convert = n * 0.1
    return convert


# ==============================================
# Normalization
# ==============================================
def normalize(word):
    if isinstance(word, str):
        word = word.lower()
        word = word.split(",", 1)[0]
    else:
        word = ""
    return word


# ==============================================
# CO2 Lookup
# ==============================================
def looking_up_co2():
    df = pd.read_csv(r"C:\Users\muthu\Documents\bfh\BFH-offenburg-burda\final_co2_emissions_only_germany.csv", encoding="cp1252")
    df = df.rename(columns={'Emissions (CO2 eq/kg)': 'Emissions'})
    df["norm"] = df["Product"].map(normalize)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    name_vecs = model.encode(df["norm"].tolist(), normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(name_vecs.shape[1])
    index.add(name_vecs)

    def embedding_lookup(name: str, addition, cos_cutoff=0.50):
        q = model.encode([normalize(name)], normalize_embeddings=True).astype("float32")
        D, I = index.search(q, 1)
        sim = float(D[0][0])
        if sim < cos_cutoff:
            return None, 0.0, addition

        idx = int(I[0][0])
        row = df.iloc[idx]
        kg_co2 = float(row["Emissions"])
        q_co2 = quantity_conversion(one_recipe_map[name])
        if q_co2 is None:
            return None, 0.0, addition
        total = kg_co2 * q_co2
        addition += total
        return row["norm"], total, addition

    addition = 0
    ingredients_list = []
    co2_list = []

    for i in list(one_recipe_map):
        p, t, a = embedding_lookup(i, addition=addition)
        addition = a
        if p is None:
            p = i
            t = 0
        ingredients_list.append(p)
        co2_list.append(t)

    return ingredients_list, co2_list, addition


# ==============================================
# Choose ingredient to swap
# ==============================================
def choosing_to_swap(co2_list, ingredients_list, total_co2):
    for i in range(len(co2_list)):
        if (co2_list[i] / total_co2) * 100 > 15:
            idx = i
    ing_switch = ingredients_list[idx]
    co2_switch = co2_list[idx]
    return ing_switch, co2_switch, idx


# ==============================================
# Get protein & carbs for swapped ingredient
# ==============================================
def getting_pro_carb(ing_switch, idx, one_recipe_map):
    df_macros_all = pd.read_csv(r"C:\Users\muthu\Documents\bfh\BFH-offenburg-burda\merged_semantic_ingredients.csv")
    df_macros = pd.DataFrame().assign(Product=df_macros_all["Product"],
                                      protein=df_macros_all["protein"],
                                      carbs=df_macros_all["carbohydrate"],
                                      units=df_macros_all["mass_kg"],
                                      co2_per_kg=df_macros_all["Emissions (CO2 eq/kg)"])

    model = SentenceTransformer("all-MiniLM-L6-v2")
    df_macros = df_macros.reset_index(drop=True)
    name_vecs = model.encode(df_macros["Product"].tolist(), normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(name_vecs.shape[1])
    index.add(name_vecs)

    def embedding_lookup(name: str, cos_cutoff=0.50):
        q = model.encode([normalize(name)], normalize_embeddings=True).astype("float32")
        D, I = index.search(q, 1)
        sim = float(D[0][0])
        if sim < cos_cutoff:
            return None, 0.0, 0.0, 0.0
        idx = int(I[0][0])
        row = df_macros.iloc[idx]
        protein = float(row["protein"])
        carbs = float(row["carbs"])
        unit = float(row["units"])
        return row["Product"], carbs, protein, unit

    product, carbs, protein, units = embedding_lookup(ing_switch)
    mu = list(one_recipe_map)[idx]
    quantity = one_recipe_map[mu]
    s = quantity.strip().lower()
    m = re.search(r'([\d.,]+)', s)
    n = float(m.group(1).replace(',', '.'))

    def convert(item, units):
        item = ((n / 1000) * units) * item
        return item

    carbs = convert(carbs, units)
    protein = convert(protein, units)
    return product, carbs, protein, units, n, df_macros


# ==============================================
# Rank alternatives
# ==============================================
def rank_alternatives(target_name: str, df: pd.DataFrame,
                      w_co2=0.4, w_protein=0.5, w_carbs=0.2, top_n=5):
    target = df[df["Product"].str.lower() == target_name.lower()]
    if target.empty:
        raise ValueError(f"{target_name} not found in dataset.")
    target = target.iloc[0]
    t_prot, t_carbs, t_co2 = target["protein"], target["carbs"], target["co2_per_kg"]

    df = df.copy()
    df["norm_co2"] = df["co2_per_kg"] / df["co2_per_kg"].max()
    df["protein_diff"] = abs(df["protein"] - t_prot) / df["protein"].max()
    df["carbs_diff"] = abs(df["carbs"] - t_carbs) / df["carbs"].max()

    df["score"] = (w_co2 * df["norm_co2"] +
                   w_protein * df["protein_diff"] +
                   w_carbs * df["carbs_diff"])

    df = df[df["Product"].str.lower() != target_name.lower()]
    df = df[df["co2_per_kg"] < t_co2]
    best = df.sort_values("score").head(top_n)
    return best[["Product", "co2_per_kg", "protein", "carbs", "score"]].reset_index(drop=True)


# ==============================================
# Replace ingredient and recompute total
# ==============================================
def replaced_lists(ingredients_list, co2_list, df_best, idx, n):
    ingredients_list[idx] = df_best["Product"][0]
    co2_list[idx] = float((n / 1000) * df_best["co2_per_kg"][0])
    return ingredients_list, co2_list, sum(co2_list)


# ==============================================
# MAIN EXECUTION
# ==============================================
if __name__ == "__main__":
    print("Translating and preparing recipe...")
    translation_ing()
    replacing_translation()

    print("Looking up CO₂ values...")
    ingredients_list, co2_list, total_co2 = looking_up_co2()

    print("Identifying high emitter...")
    ing_switch, co2_switch, idx = choosing_to_swap(co2_list, ingredients_list, total_co2)

    print(f"High emitter: {ing_switch} ({co2_switch:.2f} kg CO₂)")
    product, carbs, protein, units, n, df_macros = getting_pro_carb(ing_switch, idx, one_recipe_map)

    print(f"Protein: {protein:.2f} g | Carbs: {carbs:.2f} g")
    df_best = rank_alternatives(ing_switch, df_macros)
    print("Suggested replacements:")
    print(df_best)

    new_ingredients, new_co2, new_total = replaced_lists(ingredients_list, co2_list, df_best, idx, n)
    print("\nUpdated ingredients list and total CO₂:")
    print(new_ingredients)
    print(f"Total CO₂ after replacement: {new_total:.2f} kg")
