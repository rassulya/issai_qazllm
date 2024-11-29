import os
import csv
import json
import numpy as np
import ast
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import logging
from typing import List, Dict
from evaluate import load
import torch
from pathlib import Path


os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

code_eval = load("code_eval")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_mcq_examples(benchmark: str, lang: str) -> str:
    examples = {
        'mmlu': {
            'en': """Question: Who was the first emperor of Rome? 
Options: A) Julius Caesar B) Augustus C) Nero D) Tiberius 
Answer: B

==================================================

Question: What is the unit of electric resistance? 
Options: A) Volt B) Ampere C) Ohm D) Joule 
Answer: C

==================================================

Question: What is the derivative of f(x)=x2 with respect to x? 
Options: A) x B) 2x C) x^2 D) 2x^2 
Answer: B

==================================================""",
            'kk': """Сұрақ: Римнің алғашқы императоры кім болды? 
Нұсқалар: A) Юлий Цезарь B) Август C) Нерон D) Тиберий 
Жауап: B

==================================================

Сұрақ: Электр кедергісінің өлшем бірлігі қандай? 
Нұсқалар: A) Вольт B) Ампер C) Ом D) Джоуль 
Жауап: C

==================================================

Сұрақ: f(x)=x2 функциясының x-ке қатысты туындысы қандай? 
Нұсқалар: A) x B) 2x C) x^2 D) 2x^2 
Жауап: B

==================================================""",
            'ru': """Вопрос: Кто был первым императором Рима? 
Варианты: A) Юлий Цезарь B) Август C) Нерон D) Тиберий 
Ответ: B

==================================================

Вопрос: Какова единица измерения электрического сопротивления? 
Варианты: A) Вольт B) Ампер C) Ом D) Джоуль 
Ответ: C

==================================================

Вопрос: Какова производная функции f(x)=x2 относительно x? 
Варианты: A) x B) 2x C) x^2 D) 2x^2 
Ответ: B

=================================================="""},
        'arc': {
            'en': """Question: What causes the phases of the moon?
Options: A: the Earth's rotation B: the moon's distance from Earth C: the moon's position relative to the Earth and Sun D: clouds blocking parts of the moon
Answer: C

==================================================

Question: A metal spoon gets warmer when placed in hot soup because of which process?
Options: A: radiation B: convection C: conduction D: insulation
Answer: C

==================================================

Question: What is the main function of the human heart?
Options: A: to digest food B: to pump blood throughout the body C: to store oxygen D: to create red blood cells
Answer: B

==================================================""",
            'kk': """Сұрақ: Айдың фазаларына не себеп болады?
Нұсқалар: A: Жердің айналуы B: Айдың Жерден қашықтығы C: Айдың Жер мен Күнге қатысты орналасуы D: бұлттардың айды жартылай жабуы
Жауап: C

==================================================

Сұрақ: Металл қасық ыстық сорпаға салынғанда қандай процестің арқасында жылынады?
Нұсқалар: A: сәулелену B: конвекция C: жылу өткізгіштік D: оқшаулау
Жауап: C

==================================================

Сұрақ: Адам жүрегінің негізгі функциясы қандай?
Нұсқалар: A: тамақты қорыту B: қанды бүкіл денеге айдау C: оттегіні сақтау D: қызыл қан түйіршіктерін жасау
Жауап: B

==================================================""",
            'ru': """Вопрос: Что вызывает фазы луны?
Варианты: A: вращение Земли B: расстояние от Луны до Земли C: положение Луны относительно Земли и Солнца D: облака, закрывающие части луны
Ответ: C

==================================================

Вопрос: Металлическая ложка нагревается в горячем супе благодаря какому процессу?
Варианты: A: излучение B: конвекция C: теплопроводность D: изоляция
Ответ: C

==================================================

Вопрос: Какова основная функция человеческого сердца?
Варианты: A: переваривать пищу B: перекачивать кровь по всему телу C: хранить кислород D: создавать красные кровяные тельца
Ответ: B

=================================================="""}
    }
    return examples[benchmark][lang]

def get_math_examples(lang: str) -> str:
    math_examples = {
        'en': """Question: Carol and Jennifer are sisters from Los Angeles who love collecting signatures from celebrities. During their summer break from school, the sisters spend every afternoon collecting signatures. After five weeks, Carol and Jennifer compare their autograph books, counting up the number of signatures each sister has collected. Carol has 20 signatures in her book, and Jennifer has 44. The sisters have three more weeks of summer vacation, and they decide they want to reach 100 signatures between them by the end of the summer. How many signatures do the sisters need to collect to reach their goal?
Steps: 1. How many signatures have Carol and Jennifer collected?
2. How many signatures do the sisters need to collect?
Answer: How many signatures have Carol and Jennifer collected? ** Carol and Jennifer have already collected 20 + 44 signatures = <<20+44=64>>64 signatures.
How many signatures do the sisters need to collect? ** Since their goal is 100, they need to collect 100 - 64 signatures. 100 - 64 = <<100-64=36>>36 signatures
#### 36
Exact Answer: 36

==================================================

Question: A team of 4 painters worked on a mansion for 3/8ths of a day every day for 3 weeks. How many hours of work did each painter put in?
Steps: 1. How many hours is 3/8ths of a day?
2. How many days are in 3 weeks?
3. How many hours did each painter put in?
Answer: How many hours is 3/8ths of a day? ** There are 24 hours in a day so 3/8ths of a day is (3/8)*24 = <<3/8*24=9>>9 hours
How many days are in 3 weeks? ** One week has 7 days so 3 weeks have 7*3 = <<7*3=21>>21 days
How many hours did each painter put in? ** Each painter put in 9*21 = <<9*21=189>>189 hours of work
#### 189
Exact Answer: 189

==================================================

Question: It costs $194 per meter to repave a street. Monica's street is 150 meters long. How much more does it cost to repave Lewis' street, which is 490 meters long?
Steps: 1. What is the total cost to repave Monica's street?
2. What is the total cost to repave Lewis' street?
3. How much more does it cost to repave Lewis' street?
Answer: What is the total cost to repave Monica's street? ** Total cost to repave Monica's street is 194*150 = <<194*150=29100>>29,100 dollars.
What is the total cost to repave Lewis' street? ** Total cost to repave Lewis' street is 194*490 = <<194*490=95060>>95,060.
How much more does it cost to repave Lewis' street? ** It costs 95,060-29,100 = <<95060-29100=65960>>65,960 more dollars to repave Lewis' street.
#### 65,960
Exact Answer: 65,960

==================================================""",
        'kk': """Сұрақ: Кэрол мен Дженнифер - Лос-Анджелестен келген әпкелі-сіңлілер, олар атақты адамдардың қолтаңбаларын жинағанды жақсы көреді. Мектептегі жазғы демалыс кезінде әпкелі-сіңлілер әр түстен кейін қолтаңбалар жинайды. Бес аптадан кейін Кэрол мен Дженнифер өздерінің қолтаңба кітаптарын салыстырып, әр қарындастың жинаған қолтаңбаларын санайды. Кэролдың кітабында 20 қолтаңба, ал Дженниферде 44 қолтаңба бар. Әпкелі-сіңлілердің жазғы демалысына тағы үш апта қалды және олар жаздың соңына дейін жалпы саны 100 қолтаңбаға жеткісі келеді. Мақсатқа жету үшін әпкелі-сіңлілер қанша қолтаңба жинауы керек?
Қадамдар: 1. Кэрол мен Дженнифер қанша қолтаңба жинады?
2. Әпкелі-сіңлілер қанша қолтаңба жинауы керек?
Жауап: Кэрол мен Дженнифер қанша қолтаңба жинады? ** Кэрол мен Дженнифер 20 + 44 қолтаңба = <<20+44=64>>64 қолтаңба жинады.
Әпкелі-сіңлілер қанша қолтаңба жинауы керек? ** Мақсаттары 100 болғандықтан, олар 100 - 64 қолтаңба жинауы керек. 100 - 64 = <<100-64=36>>36 қолтаңба
#### 36
Нақты жауап: 36

==================================================

Сұрақ: 4 сырлаушыдан тұратын топ 3 апта бойы күн сайын күннің 3/8 бөлігін сарайды сырлауға жұмсады. Әр сырлаушы қанша сағат жұмыс істеді?
Қадамдар: 1. Күннің 3/8 бөлігі қанша сағат?
2. 3 аптада қанша күн бар?
3. Әр сырлаушы қанша сағат жұмыс істеді?
Жауап: Күннің 3/8 бөлігі қанша сағат? ** Бір күнде 24 сағат бар, сондықтан күннің 3/8 бөлігі (3/8)*24 = <<3/8*24=9>>9 сағат
3 аптада қанша күн бар? ** Бір аптада 7 күн бар, сондықтан 3 аптада 7*3 = <<7*3=21>>21 күн бар
Әр сырлаушы қанша сағат жұмыс істеді? ** Әр сырлаушы 9*21 = <<9*21=189>>189 сағат жұмыс істеді
#### 189
Нақты жауап: 189

==================================================

Сұрақ: Көшені қайта жөндеу құны бір метрге 194 доллар тұрады. Моника тұратын көше 150 метр. Льюис тұратын көшені қайта жөндеу қанша долларға қымбат, оның көшесі 490 метр?
Қадамдар: 1. Моника тұратын көшені қайта жөндеудің жалпы құны қанша?
2. Льюис тұратын көшені қайта жөндеудің жалпы құны қанша?
3. Льюис тұратын көшені қайта жөндеу қанша долларға қымбат?
Жауап: Моника тұратын көшені қайта жөндеудің жалпы құны қанша? ** Моника тұратын көшені қайта жөндеудің жалпы құны 194*150 = <<194*150=29100>>29,100 доллар.
Льюис тұратын көшені қайта жөндеудің жалпы құны қанша? ** Льюис тұратын көшені қайта жөндеудің жалпы құны 194*490 = <<194*490=95060>>95,060 доллар.
Льюис тұратын көшені қайта жөндеу қанша долларға қымбат? ** Льюис тұратын көшені қайта жөндеу 95,060-29,100 = <<95060-29100=65960>>65,960 долларға қымбат.
#### 65,960
Нақты жауап: 65,960

==================================================""",
        'ru': """Вопрос: Кэрол и Дженнифер - сестры из Лос-Анджелеса, которые любят собирать подписи знаменитостей. Во время летних каникул сестры каждый день после обеда собирают подписи. Через пять недель Кэрол и Дженнифер сравнивают свои книги автографов, подсчитывая количество подписей, собранных каждой сестрой. У Кэрол в книге 20 подписей, а у Дженнифер 44. У сестер осталось еще три недели летних каникул, и они решили, что хотят достичь 100 подписей между ними к концу лета. Сколько подписей нужно собрать сестрам, чтобы достичь своей цели?
Шаги: 1. Сколько подписей собрали Кэрол и Дженнифер?
2. Сколько подписей нужно собрать сестрам?
Ответ: Сколько подписей собрали Кэрол и Дженнифер? ** Кэрол и Дженнифер уже собрали 20 + 44 подписи = <<20+44=64>>64 подписи.
Сколько подписей нужно собрать сестрам? ** Поскольку их цель - 100, им нужно собрать 100 - 64 подписи. 100 - 64 = <<100-64=36>>36 подписей
#### 36
Точный ответ: 36

==================================================

Вопрос: Команда из 4 маляров работала над особняком по 3/8 дня каждый день в течение 3 недель. Сколько часов работы выполнил каждый маляр?
Шаги: 1. Сколько часов составляет 3/8 дня?
2. Сколько дней в 3 неделях?
3. Сколько часов проработал каждый маляр?
Ответ: Сколько часов составляет 3/8 дня? ** В дне 24 часа, поэтому 3/8 дня это (3/8)*24 = <<3/8*24=9>>9 часов
Сколько дней в 3 неделях? ** В одной неделе 7 дней, поэтому в 3 неделях 7*3 = <<7*3=21>>21 день
Сколько часов проработал каждый маляр? ** Каждый маляр проработал 9*21 = <<9*21=189>>189 часов
#### 189
Точный ответ: 189

==================================================

Вопрос: Перекладка улицы стоит 194 доллара за метр. Улица Моники длиной 150 метров. На сколько дороже перекладка улицы Льюиса, которая имеет длину 490 метров?
Шаги: 1. Какова общая стоимость перекладки улицы Моники?
2. Какова общая стоимость перекладки улицы Льюиса?
3. На сколько дороже перекладка улицы Льюиса?
Ответ: Какова общая стоимость перекладки улицы Моники? ** Общая стоимость перекладки улицы Моники составляет 194*150 = <<194*150=29100>>29,100 долларов.
Какова общая стоимость перекладки улицы Льюиса? ** Общая стоимость перекладки улицы Льюиса составляет 194*490 = <<194*490=95060>>95,060 долларов.
На сколько дороже перекладка улицы Льюиса? ** Перекладка улицы Льюиса дороже на 95,060-29,100 = <<95060-29100=65960>>65,960 долларов.
#### 65,960
Точный ответ: 65,960

=================================================="""}
    return math_examples[lang]

# Add these functions after the imports and before the other functions

def get_one_shot_example_en():
    return """Question: def median(l: list):
    \"\"\"Return median of elements in the list l.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    \"\"\"

Answer:
Your code should be inside these symbols:
```
def median(l: list):
    l = sorted(l)
    if len(l) % 2 == 1:
        return l[len(l) // 2]
    else:
        return (l[len(l) // 2 - 1] + l[len(l) // 2]) / 2.0
```
=================================================="""

def get_one_shot_example_kk():
    return """Сұрақ: def median(l: list):
    \"\"\"Тізімдегі элементтердің медианасын қайтарады.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    \"\"\"

Жауап:
Сіздің кодыңыз осы таңбалардың ішінде болуы керек:
```
def median(l: list):
    l = sorted(l)
    if len(l) % 2 == 1:
        return l[len(l) // 2]
    else:
        return (l[len(l) // 2 - 1] + l[len(l) // 2]) / 2.0
```
=================================================="""

def get_one_shot_example_ru():
    return """Вопрос: def median(l: list):
    \"\"\"Возвращает медиану элементов в списке.
    >>> median([3, 1, 2, 4, 5])
    3
    >>> median([-10, 4, 6, 1000, 10, 20])
    15.0
    \"\"\"

Ответ:
Ваш код должен быть внутри этих символов:
```
def median(l: list):
    l = sorted(l)
    if len(l) % 2 == 1:
        return l[len(l) // 2]
    else:
        return (l[len(l) // 2 - 1] + l[len(l) // 2]) / 2.0
```
=================================================="""

# Rename the old create_humaneval_prompt to create_humaneval_prompt_old

# Add your new humaneval_prompt function
def create_humaneval_prompt(row: Dict, lang: str = 'en') -> str:
    # Main instruction in English
    main_instruction = "You are my intelligent coding assistant."
    
    # Language-specific components
    examples = {
        'en': get_one_shot_example_en(),
        'kk': get_one_shot_example_kk(),
        'ru': get_one_shot_example_ru()
    }
    
    question_prefix = {
        'en': "You have this Python function:",
        'kk': "Сізде мынадай Python функциясы бар:",
        'ru': "У вас есть такая функция Python:"
    }
    
    code_instruction = {
        'en': "Your code should be inside these symbols: ```\nGive me the code without any explanation and extra symbols.",
        'kk': "Сіздің кодыңыз осы таңбалардың ішінде болуы керек: ```\nМаған кодты қосымша түсініктемелер мен таңбаларсыз беріңіз.",
        'ru': "Ваш код должен быть внутри этих символов: ```\nДайте мне код без объяснений и дополнительных символов."
    }
    
    answer_label = {
        'en': "Answer:",
        'kk': "Жауап:",
        'ru': "Ответ:"
    }
    
    prompt = f"""{main_instruction}

Here is an example problem and its solution:

{examples[lang]}

{question_prefix[lang]}

{row['prompt']}

{code_instruction[lang]}
{answer_label[lang]} """
    return prompt

def extract_code_from_markdown(text: str) -> str:
    """Extract code from between ``` markers and clean it up."""
    pattern = r"```(?:python\n|\n)?(.+?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        code = matches[0].strip()
    else:
        code = text.strip()
    
    func_pattern = r"(def\s+.*?(?:\n|$).*)"
    func_matches = re.findall(func_pattern, code, re.DOTALL)
    
    if func_matches:
        func_code = func_matches[0].strip()
        while func_code and func_code[-1] in ['"', "'", ',', '.', '\n']:
            func_code = func_code[:-1]
        return func_code
    
    return code.strip()

def create_humaneval_prompt_old(row: Dict) -> str:
    return f"""You are my intelligent coding assistant.
You have this Python function:

{row['prompt']}

Finish this function.
Give me the code without any explanation and extra symbols."""

def create_mcq_prompt_old(row: Dict, benchmark: str, lang: str) -> str:
    separator = ': ' if benchmark == 'arc' else ') '
    
    prompts = {
        'en': {
            'question': 'Question',
            'options': 'Options',
            'answer': 'Answer'
        },
        'kk': {
            'question': 'Сұрақ',
            'options': 'Нұсқалар',
            'answer': 'Жауап'
        },
        'ru': {
            'question': 'Вопрос',
            'options': 'Варианты',
            'answer': 'Ответ'
        }
    }

    return f"""You are an intelligent AI assistant specialized in answering multiple choice questions. For each question, you will be given four options (A, B, C, D). Your task is to analyze the question and options carefully, then provide only the letter of the correct answer.

Here are some example problems and their solutions:

{get_mcq_examples(benchmark, lang)}

{prompts[lang]['question']}: {row['question']}
{prompts[lang]['options']}: A{separator}{row['A']} B{separator}{row['B']} C{separator}{row['C']} D{separator}{row['D']}
{prompts[lang]['answer']}:"""

def create_mcq_prompt(row: Dict, benchmark: str, lang: str) -> str:
    """Create zero-shot MCQ prompt for ARC and MMLU"""
    prompts = {
        'en': f"""The following question is presented for you to answer:
Question: {row['question']}
Choose the correct answer from the options listed below:
A: {row['A']}
B: {row['B']}
C: {row['C']}
D: {row['D']}
Return your answer as a single word, in the following format: X, where X is the letter indicating the correct option. Do not explain your answer.
Answer: """,
        'kk': f"""Келесі сұраққа жауап беруіңіз керек:
Сұрақ: {row['question']}
Төменде берілген нұсқалардан дұрыс жауапты таңдаңыз:
A: {row['A']}
B: {row['B']}
C: {row['C']}
D: {row['D']}
Жауабыңызды бір сөзбен беріңіз, келесі форматта: X, мұндағы X дұрыс нұсқаны көрсететін әріп. Жауабыңызды түсіндірмеңіз.
Жауап: """,
        'ru': f"""Вам предлагается ответить на следующий вопрос:
Вопрос: {row['question']}
Выберите правильный ответ из представленных ниже вариантов:
A: {row['A']}
B: {row['B']}
C: {row['C']}
D: {row['D']}
Дайте ответ одним словом в следующем формате: X, где X - буква, указывающая на правильный вариант. Не объясняйте свой ответ.
Ответ: """
    }
    return prompts[lang]

def create_winogrande_prompt(row: Dict, lang: str) -> str:
    prompts = {
        'en': f"""You are given a sentence with one missing word, indicated by an underscore. Read the sentence and choose the appropriate option from the two provided to fill in the blank.
Sentence: {row['sentence']}
Option 1: {row['option1']}
Option 2: {row['option2']}
Return your answer as a single word, in the following format: X, where X is the number (1 or 2) corresponding to the correct option. Do not include any explanation.
Answer: """,
        'kk': f"""Сізге бір сөз жетіспейтін сөйлем берілген, ол астын сызумен көрсетілген. Сөйлемді оқып, берілген екі нұсқадан бос орынға сәйкес келетін нұсқаны таңдаңыз.
Сөйлем: {row['sentence']}
1-нұсқа: {row['option1']}
2-нұсқа: {row['option2']}
Жауабыңызды X форматында бір сөзбен беріңіз, мұндағы X дұрыс нұсқаға сәйкес келетін сан (1 немесе 2). Түсіндірме қоспаңыз.
Жауап: """,
        'ru': f"""Вам дано предложение с одним пропущенным словом, обозначенным подчеркиванием. Прочитайте предложение и выберите подходящий вариант из двух предложенных для заполнения пропуска.
Предложение: {row['sentence']}
Вариант 1: {row['option1']}
Вариант 2: {row['option2']}
Дайте ответ одним словом в следующем формате: X, где X - это число (1 или 2), соответствующее правильному варианту. Не включайте объяснение.
Ответ: """
    }
    return prompts[lang]

def create_hellaswag_prompt(row: Dict, lang: str) -> str:
    prompts = {
        'en': f"""Please complete the sentence given below:
'{row['ctx']}'...
Choose the correct ending from the options listed below:
1: '{row['option1']}'
2: '{row['option2']}'
3: '{row['option3']}'
4: '{row['option4']}'
Return your answer as a single word, in the following format: X, where X is the number indicating the correct option. Do not explain your answer.
Answer: """,
        'kk': f"""Төмендегі берілген сөйлемді аяқтаңыз:
'{row['ctx']}'...
Төменде берілген нұсқалардан дұрыс аяқталуын таңдаңыз:
1: '{row['option1']}'
2: '{row['option2']}'
3: '{row['option3']}'
4: '{row['option4']}'
Жауабыңызды X форматында бір сөзбен беріңіз, мұндағы X дұрыс нұсқаны көрсететін сан. Жауабыңызды түсіндірмеңіз.
Жауап: """,
        'ru': f"""Пожалуйста, закончите предложение, данное ниже:
'{row['ctx']}'...
Выберите правильное окончание из перечисленных вариантов:
1: '{row['option1']}'
2: '{row['option2']}'
3: '{row['option3']}'
4: '{row['option4']}'
Дайте ответ одним словом в следующем формате: X, где X - это число, указывающее на правильный вариант. Не объясняйте свой ответ.
Ответ: """
    }
    return prompts[lang]

def create_drop_prompt(row: Dict, lang: str) -> str:
    prompts = {
        'en': f"""You are intelligent assistant in answering to questions from the given passage.
You will be given a passage and question. Read the passage first and look for the answer for the question.
Give the exact answer only without an explanation.
Give the answer in the given language of the passage and question.
If it is in english, your answer should be in english.
If it is in kazakh, your answer should be in kazakh.
If it is in russian, your answer should be in russian.

Read the following passage and answer the question:
Passage: {row['passage']}
Question: {row['question']}
Return your answer only. Do not include any explanation, just the final answer.
Answer: """,
        'kk': f"""You are intelligent assistant in answering to questions from the given passage.
You will be given a passage and question. Read the passage first and look for the answer for the question.
Give the exact answer only without an explanation.
Give the answer in the given language of the passage and question.
If it is in english, your answer should be in english.
If it is in kazakh, your answer should be in kazakh.
If it is in russian, your answer should be in russian.

Төмендегі мәтінді оқып, сұраққа жауап бер:
Мәтін: {row['passage']}
Сұрақ: {row['question']}
Тек жауапты бер. Ешқандай түсініктеме қосудың қажеті жоқ.
Жауабы: """,
        'ru': f"""You are intelligent assistant in answering to questions from the given passage.
You will be given a passage and question. Read the passage first and look for the answer for the question.
Give the exact answer only without an explanation.
Give the answer in the given language of the passage and question.
If it is in english, your answer should be in english.
If it is in kazakh, your answer should be in kazakh.
If it is in russian, your answer should be in russian.

Прочитайте следующий отрывок и ответьте на вопрос:
Отрывок: {row['passage']}
Вопрос: {row['question']}
Дайте только ответ без объяснений.
Ответ: """
    }
    return prompts[lang]

def create_math_prompt(row: Dict, lang: str) -> str:
    prompts = {
        'en': f"""You are intelligent assistant in solving mathematical questions.
You are given steps which are intermediate questions to calculate the final answer. Solve this questions first, and then calculate the final answer.
Give the exact answer only without an explanation.
Do not show me the intermediate calculations, give only the final answer.
Do not generate anything else, only the final answer is needed that comes after.
I need only the the number that comes after exact answer.

Here are some example problems and their solutions:

{get_math_examples('en')}

Question: {row['question']}
Steps: 
{row['hints']}
Return your answer as a single number.
Answer: """,
        'kk': f"""You are intelligent assistant in solving mathematical questions.
You are given steps which are intermediate questions to calculate the final answer. Solve this questions first, and then calculate the final answer.
Give the exact answer only without an explanation.
Do not show me the intermediate calculations, give only the final answer.
Do not generate anything else, only the final answer is needed.
I need only the the number that comes after нақты жауап.

Here are some example problems and their solutions:

{get_math_examples('kk')}

Сұрақ: {row['question']}
Қадамдар:
{row['hints']}
Жауабыңызды бір сан ретінде беріңіз.
Жауап: """,
        'ru': f"""You are intelligent assistant in solving mathematical questions.
You are given steps which are intermediate questions to calculate the final answer. Solve this questions first, and then calculate the final answer.
Give the exact answer only without an explanation.
Do not show me the intermediate calculations, give only the final answer.
Do not generate anything else, only the final answer is needed.
I need only the the number that comes after точный ответ.

Here are some example problems and their solutions:

{get_math_examples('ru')}

Вопрос: {row['question']}
Шаги:
{row['hints']}
Дайте ответ в виде одного числа.
Ответ: """
    }
    return prompts[lang]

def extract_mcq_answer(text: str) -> str:
    text = str(text).strip().upper()
    
    answer_patterns = [
        r'(?i)answer[\s:\n\.,]+([ABCD])',
        r'(?i)ответ[\s:\n\.,]+([ABCD])',
        r'(?i)жауап[\s:\n\.,]+([ABCD])'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    matches = re.findall(r'\b[ABCD]\b', text)
    return matches[0] if matches else None

def extract_drop_answer(result: str) -> str:
    return str(result).strip()

def extract_math_answer(text: str, lang: str = 'en') -> str:
    text = str(text).strip()
    
    markers = {
        'en': 'Exact Answer:',
        'kk': 'Нақты жауап:',
        'ru': 'Точный ответ:'
    }
    
    marker = markers.get(lang, markers['en'])
    
    try:
        # First try to extract by language-specific marker
        if marker in text:
            after_marker = text.split(marker, 1)[1].strip()
            numbers = re.findall(r'[-+]?\d*\.?\d+', after_marker)
            if numbers:
                return numbers[0]
        
        # Then try to extract by #### delimiter
        if '####' in text:
            after_delimiter = text.split('####', 1)[1].strip()
            # Remove any currency symbols and whitespace
            cleaned_text = re.sub(r'[^\d.-]', '', after_delimiter)
            if cleaned_text:
                return cleaned_text
        
        # If no delimiters found, try to find any numbers
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if numbers:
            return numbers[0]
        
        return text
        
    except Exception as e:
        logging.error(f"Error extracting math answer: {e}")
        return text

# def extract_math_answer(text: str, lang: str = 'en') -> str:
#     text = str(text).strip()
    
#     markers = {
#         'en': 'Exact Answer:',
#         'kk': 'Нақты жауап:',
#         'ru': 'Точный ответ:'
#     }
    
#     marker = markers.get(lang, markers['en'])
    
#     try:
#         if marker in text:
#             after_marker = text.split(marker, 1)[1].strip()
#             numbers = re.findall(r'[-+]?\d*\.?\d+', after_marker)
#             if numbers:
#                 return numbers[0]
        
#         numbers = re.findall(r'[-+]?\d*\.?\d+', text)
#         if numbers:
#             return numbers[0]
        
#         return text
#     except Exception as e:
#         logging.error(f"Error extracting math answer: {e}")
#         return text

def evaluate_drop_answer(result: str, answers: List, lang: str = 'en') -> int:
    result_str = str(result).strip()
    
    def normalize_text(text, lang):
        text = text.strip()
        if lang in ['kk', 'ru']:
            text = re.sub(r'(\d+)\.(\d+)', r'\1,\2', text)
        return text
    
    def create_pattern(answer):
        answer_escaped = re.escape(str(answer).strip())
        return fr'\n?\b{answer_escaped}\b[.,\s%\n]*'

    def are_words_similar(word1, word2):
        w1 = re.sub(r'[.,\s%\n]+$', '', word1.lower())
        w2 = re.sub(r'[.,\s%\n]+$', '', word2.lower())
        
        if w1 in w2 or w2 in w1:
            return True
        
        if len(w1) > 2 and len(w2) > 2:
            if w1[:-1] == w2[:-1]:
                return True
        
        return False
    
    count = 0
    normalized_result = normalize_text(result_str, lang)
    
    for answer in answers:
        answer_str = str(answer).strip()
        normalized_answer = normalize_text(answer_str, lang)
        
        if normalized_result == normalized_answer:
            count += 1
            continue
        
        pattern = create_pattern(normalized_answer)
        if re.search(pattern, normalized_result, re.MULTILINE):
            count += 1
            continue
            
        if lang in ['kk', 'ru'] and '.' in answer_str:
            alt_answer = answer_str.replace('.', ',')
            pattern = create_pattern(alt_answer)
            if re.search(pattern, normalized_result, re.MULTILINE):
                count += 1
                continue

        if are_words_similar(normalized_result, normalized_answer):
            count += 1
            continue
    
    return min(count, 1)

def evaluate_math_answer(result: str, answers: List, lang: str = 'en') -> int:
    """Evaluate math answers with special handling for numerical comparisons."""
    result_str = str(result).strip()
    
    def normalize_text(text, lang):
        text = text.strip()
        if lang in ['en', 'kk', 'ru']:
            text = re.sub(r'(\d+)\.(\d+)', r'\1,\2', text)
        return text
    
    def are_numbers_equal(num1, num2):
        try:
            n1 = float(str(num1).replace(',', '').replace(' ', '').replace('،', '').rstrip('.,'))
            n2 = float(str(num2).replace(',', '').replace(' ', '').replace('،', '').rstrip('.,'))
            return abs(n1 - n2) < 1e-10
        except ValueError:
            return False
    
    normalized_result = normalize_text(result_str, lang)
    
    for answer in answers:
        if are_numbers_equal(normalized_result, answer):
            return 1
            
    return 0

class EvaluationDataset(Dataset):
    def __init__(self, dataset: List[Dict], benchmark: str, lang: str):
        self.dataset = dataset
        self.benchmark = benchmark
        self.lang = lang

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.benchmark == 'humaneval':
            prompt = create_humaneval_prompt_old(item)
            return {
                'prompt': prompt,
                'task_id': item['task_id'],
                'original_prompt': item['prompt'],
                'lang': self.lang
            }
        elif self.benchmark == 'gsm8k':
            prompt = create_math_prompt(item, self.lang)
            return {
                'prompt': prompt,
                'question': item['question'],
                'hints': item.get('hints', ''),
                'answer': item['answer']
            }
        elif self.benchmark == 'drop':
            prompt = create_drop_prompt(item, self.lang)
            return {
                'prompt': prompt,
                'query_id': item['query_id'],
                'passage': item['passage'],
                'question': item['question'],
                'answers': item['answers']
            }
        elif self.benchmark == 'winogrande':
            prompt = create_winogrande_prompt(item, self.lang)
            return {
                'prompt': prompt,
                'sentence': item['sentence'],
                'option1': item['option1'],
                'option2': item['option2'],
                'answer': str(item['answer'])
            }
        elif self.benchmark == 'hellaswag':
            prompt = create_hellaswag_prompt(item, self.lang)
            return {
                'prompt': prompt,
                'ctx': item['ctx'],
                'option1': item['option1'],
                'option2': item['option2'],
                'option3': item['option3'],
                'option4': item['option4'],
                'answer': str(item['answer'])
            }
        else:
            prompt = create_mcq_prompt(item, self.benchmark, self.lang)
            return {
                'prompt': prompt,
                'question': item['question'],
                'A': item['A'],
                'B': item['B'],
                'C': item['C'],
                'D': item['D'],
                'answer': str(item['answer']).strip().upper()
            }

def load_dataset(file_path: str, data_portion: int = 100) -> List[Dict]:
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        
        if 'answer' in df.columns:  # For MCQ datasets
            df['answer'] = df['answer'].astype(str).str.strip().str.upper()
            invalid_answers = df[~df['answer'].isin(['A', 'B', 'C', 'D'])]
            if not invalid_answers.empty:
                logging.warning(f"Invalid answers found in {file_path}:")
                logging.warning(invalid_answers)
        
        data = df.to_dict('records')
        
        if data_portion < 100:
            num_samples = int(len(data) * (data_portion / 100))
            return random.sample(data, num_samples)
        return data
        
    except Exception as e:
        logging.error(f"Error loading dataset {file_path}: {str(e)}")
        raise

def evaluate_model(llm: LLM, dataset: List[Dict], benchmark: str, lang: str, max_tokens: int, batch_size: int = 32) -> List[Dict]:
    eval_dataset = EvaluationDataset(dataset, benchmark, lang)
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    results = []
    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=max_tokens,
        stop=['=================================================='] if benchmark != 'gsm8k' else None
    )
    
    for batch in tqdm(dataloader, desc=f"Evaluating {benchmark.upper()} {lang}"):
        prompts = batch['prompt']
        outputs = llm.generate(prompts, sampling_params)
        generated = [output.outputs[0].text.strip() for output in outputs]
        
        if benchmark == 'humaneval':
            for i, gen in enumerate(generated):
                clean_code = extract_code_from_markdown(gen)
                results.append({
                    'task_id': batch['task_id'][i],
                    'original_prompt': batch['original_prompt'][i],
                    'generated': clean_code,
                    'language': lang,
                    'status': 'pending'  # Will be updated after pass@k calculation
                })
        elif benchmark == 'gsm8k':
            for i, gen in enumerate(generated):
                answer = batch['answer'][i]
                extracted_answer = extract_math_answer(gen, lang)
                
                try:
                    answers = [float(answer)]
                except (ValueError, TypeError):
                    answers = [answer]
                
                is_correct = evaluate_math_answer(extracted_answer, answers, lang) > 0
                
                results.append({
                    'question': batch['question'][i],
                    'hints': batch['hints'][i],
                    'answer': batch['answer'][i],
                    'generated': gen,
                    'extracted_answer': extracted_answer,
                    'language': lang,
                    'status': 'correct' if is_correct else 'incorrect'
                })
        elif benchmark == 'drop':
            for i, gen in enumerate(generated):
                answers = ast.literal_eval(batch['answers'][i])
                is_correct = evaluate_drop_answer(gen, answers, lang) > 0
                
                results.append({
                    'query_id': batch['query_id'][i],
                    'passage': batch['passage'][i],
                    'question': batch['question'][i],
                    'answers': batch['answers'][i],
                    'generated': gen,
                    'language': lang,
                    'status': 'correct' if is_correct else 'incorrect'
                })
        elif benchmark == 'winogrande':
            for i, gen in enumerate(generated):
                # Extract just the number from the generated text
                extracted_answer = ''.join(filter(str.isdigit, gen))[:1]  # Take first digit
                is_correct = extracted_answer == batch['answer'][i]
                
                results.append({
                    'sentence': batch['sentence'][i],
                    'option1': batch['option1'][i],
                    'option2': batch['option2'][i],
                    'correct_answer': batch['answer'][i],
                    'generated': gen,
                    'extracted_answer': extracted_answer,
                    'language': lang,
                    'status': 'correct' if is_correct else 'incorrect'
                })
        elif benchmark == 'hellaswag':
            for i, gen in enumerate(generated):
                # Extract just the number from the generated text
                extracted_answer = ''.join(filter(str.isdigit, gen))[:1]  # Take first digit
                is_correct = extracted_answer == batch['answer'][i]
                
                results.append({
                    'ctx': batch['ctx'][i],
                    'option1': batch['option1'][i],
                    'option2': batch['option2'][i],
                    'option3': batch['option3'][i],
                    'option4': batch['option4'][i],
                    'correct_answer': batch['answer'][i],
                    'generated': gen,
                    'extracted_answer': extracted_answer,
                    'language': lang,
                    'status': 'correct' if is_correct else 'incorrect'
                })        
        else:
            for i, gen in enumerate(generated):
                extracted_answer = extract_mcq_answer(gen)
                is_correct = extracted_answer == batch['answer'][i] if extracted_answer else False
                
                results.append({
                    'question': batch['question'][i],
                    'A': batch['A'][i],
                    'B': batch['B'][i],
                    'C': batch['C'][i],
                    'D': batch['D'][i],
                    'correct_answer': batch['answer'][i],
                    'generated': gen,
                    'extracted_answer': extracted_answer,
                    'language': lang,
                    'status': 'correct' if is_correct else 'incorrect'
                })

    return results

def calculate_pass_at_k(df: pd.DataFrame, df_generated: pd.DataFrame) -> float:
    pass_k = np.zeros(df.shape[0])

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        tests = ast.literal_eval(row['tests'])
        test_pass = []
        for test in tests:
            test_cases = [test]
            candidates = [[df_generated['generated'].iloc[idx]]]
            pass_at_k, _ = code_eval.compute(references=test_cases, predictions=candidates, k=[1])
            test_pass.append(pass_at_k['pass@1'])
        pass_k[idx] = sum(test_pass) / len(test_pass)

    return np.mean(pass_k)

def save_results(results: List[Dict], benchmark: str, model_name: str):
    try:
        output_path = Path("data/evaluation") 
        output_path.parent.resolve().mkdir(parents=True, exist_ok=True)
        output_file = os.path.join(output_dir, f"{benchmark}_generated_{model_name}.csv")
        
        df = pd.DataFrame(results)
        
        # Define columns for each benchmark type
        if benchmark == 'drop':
            columns = ['query_id', 'passage', 'question', 'answers', 'generated', 'status', 'language']
        elif benchmark == 'gsm8k':
            columns = ['question', 'hints', 'answer', 'generated', 'extracted_answer', 'status', 'language']
        elif benchmark == 'humaneval':
            columns = ['task_id', 'original_prompt', 'generated', 'status', 'language']
        elif benchmark == 'winogrande':
            columns = ['sentence', 'option1', 'option2', 'correct_answer', 'generated', 'extracted_answer', 'status', 'language']
        elif benchmark == 'hellaswag':
            columns = ['ctx', 'option1', 'option2', 'option3', 'option4', 'correct_answer', 'generated', 'extracted_answer', 'status', 'language']
        else:  # for MMLU and ARC
            columns = ['question', 'A', 'B', 'C', 'D', 'correct_answer', 'generated', 'extracted_answer', 'status', 'language']
        
        # Select only columns that exist in the DataFrame
        columns = [col for col in columns if col in df.columns]
        
        df = df[columns]
        df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"Results saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")
        raise

def save_accuracy_metrics(accuracy_results: Dict, model_name: str, output_dir: str = "results"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"accuracy_metrics_{model_name}.json")
        
        # Restructure the results according to the new format
        formatted_results = {
            "model_name": model_name,
            "results": {
                "en": {},
                "kk": {},
                "ru": {}
            }
        }
        
        # Populate the results
        for key, value in accuracy_results.items():
            benchmark, lang = key.split('_')
            formatted_results["results"][lang][benchmark] = {"acc": value}
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, indent=2)
        logging.info(f"Accuracy metrics saved to {output_file}")
        
    except Exception as e:
        logging.error(f"Error saving accuracy metrics: {str(e)}")
        raise

# def get_model_name_from_path(model_path: str) -> str:
#     base_name = os.path.basename(model_path)
#     model_name = os.path.splitext(base_name)[0]
#     return model_name

def main(
    model_path: str,
    model_name: str,
    benchmarks: List[str] = ['arc', 'mmlu', 'drop', 'gsm8k', 'humaneval', 'hellaswag', 'winogrande'],
    languages: List[str] = ['en', 'kk', 'ru'],
    batch_size: int = 32,
    tensor_parallel_size: int = 4,
    data_portion: int = 100,
    max_tokens_mmlu: int = 15,
    max_tokens_arc: int = 30,
    max_tokens_drop: int = 40,
    max_tokens_gsm8k: int = 512,
    max_tokens_humaneval: int = 128,
    max_tokens_hellaswag: int = 15,
    max_tokens_winogrande: int = 15,
    output_dir: str = "results",
    is_local_model = False,
    datasets = "data/benchmark/datasets_v2"
):
    try:
        logging.info(f"Loading model: {model_path}")
        if is_local_model:
            model_path = Path(model_path).expanduser().resolve().absolute()
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.90,
        )
        
        accuracy_metrics = {}
        
        # Map benchmarks to their respective token limits
        max_tokens_map = {
            'mmlu': max_tokens_mmlu,
            'arc': max_tokens_arc,
            'drop': max_tokens_drop,
            'gsm8k': max_tokens_gsm8k,
            'humaneval': max_tokens_humaneval,
            'hellaswag': max_tokens_hellaswag,
            'winogrande': max_tokens_winogrande
        }
        
        for benchmark in benchmarks:
            benchmark_results = []
            print(benchmark)
            max_tokens = max_tokens_map[benchmark]
            
            for lang in languages:
                try:
                    # Special handling for different dataset paths
                    if benchmark == 'gsm8k':
                        dataset_path = f"{datasets}/gsm8k_{lang}_v2.csv"
                    elif benchmark == 'humaneval':
                        dataset_path = f"{datasets}/humaneval_{lang}.csv"
                    elif benchmark == "arc":
                        dataset_path = f"{datasets}/arc_{lang}_v2.csv"
                    else:
                        dataset_path = f"{datasets}/{benchmark}_{lang}.csv"
                        
                    logging.info(f"Processing {benchmark.upper()} dataset for {lang} with {max_tokens} max tokens")
                    
                    dataset = load_dataset(dataset_path, data_portion)
                    results = evaluate_model(llm, dataset, benchmark, lang, max_tokens, batch_size)
                    benchmark_results.extend(results)
                    
                    # Calculate accuracy based on benchmark type
                    if benchmark == 'humaneval':
                        df = pd.read_csv(dataset_path)
                        if data_portion < 100:
                            df = df.iloc[:len(results)]
                        df_generated = pd.DataFrame(results)
                        accuracy = calculate_pass_at_k(df, df_generated)
                    else:
                        correct = sum(1 for r in results if r['status'] == 'correct')
                        total = len(results)
                        accuracy = correct / total if total > 0 else 0
                    
                    accuracy_metrics[f'{benchmark}_{lang}'] = accuracy
                    
                    logging.info(f"Evaluation complete for {benchmark} {lang}")
                    logging.info(f"Accuracy: {accuracy}")
                    
                except Exception as e:
                    logging.error(f"Error processing {benchmark} for {lang}: {str(e)}")
                    continue
            
            save_results(benchmark_results, benchmark, model_name, output_dir)
        
        save_accuracy_metrics(accuracy_metrics, model_name, output_dir)
        logging.info("All evaluations complete. Results saved.")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_NAME = "Llama-3.2-1B-Instruct"
    BENCHMARKS = ["gsm8k", "arc", "mmlu", "drop", "hellaswag", "humaneval", "winogrande"], 
    LANGUAGES = [ "kk", "ru", "en"]
    BATCH_SIZE = 64
    TENSOR_PARALLEL_SIZE = 8
    DATA_PORTION = 100
    MAX_TOKENS_MMLU = 15
    MAX_TOKENS_ARC = 30
    MAX_TOKENS_DROP = 40
    MAX_TOKENS_GSM8K = 512
    MAX_TOKENS_HUMANEVAL = 512  
    MAX_TOKENS_HELLASWAG = 15 
    MAX_TOKENS_WINOGRANDE = 15
    IS_LOCAL = False
    
    main(
        model_path=MODEL_PATH,
        model_name=MODEL_NAME,
        benchmarks=BENCHMARKS,
        languages=LANGUAGES,
        batch_size=BATCH_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        data_portion=DATA_PORTION,
        max_tokens_mmlu=MAX_TOKENS_MMLU,
        max_tokens_arc=MAX_TOKENS_ARC,
        max_tokens_drop=MAX_TOKENS_DROP,
        max_tokens_gsm8k=MAX_TOKENS_GSM8K,
        max_tokens_humaneval=MAX_TOKENS_HUMANEVAL,
        max_tokens_hellaswag=MAX_TOKENS_HELLASWAG,
        max_tokens_winogrande=MAX_TOKENS_WINOGRANDE,
        is_local_model = False
    )
