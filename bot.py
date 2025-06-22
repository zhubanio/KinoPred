# файл bot.py
import os
import pickle
import random
import re
import time
from io import BytesIO
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
import requests
import json
from typing import Optional
from telegram import Message

from pipeline.train_and_ensemble import train_and_ensemble

os.environ["JOBLIB_TEMP_FOLDER"] = r"C:\tmp\cachedir"


def escape_markdown_v2(text: str) -> str:
    """Надежно экранирует все специальные символы для Telegram MarkdownV2."""
    # Список всех зарезервированных символов в MarkdownV2
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


# noinspection PyUnusedLocal
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    :param update: Update
    :type context: ContextTypes.DEFAULT_TYPE
    """
    global CHAT_USERS
    global markup
    global markup
    if CHAT_USERS:
        keyboard = list()
        keyboard.append([InlineKeyboardButton("Да", callback_data=f"start_over")])
        keyboard.append([InlineKeyboardButton("Нет", callback_data=f"go_on")])
        markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Вы уверены, что хотите перезапустить чат?"
                                        " Список выбранных пользователей будет очищен, "
                                        "но сами пользователи останутся сохранены.", reply_markup=markup)
        if CURRENT_USER:
            CHAT_USERS[CURRENT_USER]["state"] = "reboot"

    else:
        await update.message.reply_text("Привет, я KinoPred! Введите id пользователя",
                                        reply_markup=instructions_markup)


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CHAT_USERS
    global CURRENT_USER
    # global user
    if CURRENT_USER:
        if CHAT_USERS[CURRENT_USER]["state"] == "API":
            await get_api(update, context)
        elif CHAT_USERS[CURRENT_USER]["state"] == "movie_id":
            await print_prediction(update, context)
        else:
            await go_on(update, context)
    else:
        await get_id(update, context)


async def get_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CHAT_USERS
    global CURRENT_USER
    global markup
    user_input = update.message.text
    if user_input.isdigit():
        USER_ID = int(user_input)
        if USER_ID == 1:
            await update.message.reply_text("Это профиль самого Кинопоиска. Введите id своего.",
                                            reply_markup=instructions_markup)
        else:
            with open("users_database/users.json", "r", encoding="utf-8") as f:
                users = json.load(f)
            if str(USER_ID) in users:
                await update.message.reply_text("Пользователь найден в базе данных")
                CURRENT_USER = USER_ID
                await update.message.reply_text(f"С возвращением, {users[str(USER_ID)]['name']}")
                CHAT_USERS[CURRENT_USER] = users[str(USER_ID)]
                CHAT_USERS[CURRENT_USER]["state"] = "default"
                await go_on(update, context)
            else:
                k = 0
                url = f"https://www.kinopoisk.ru/user/{USER_ID}/"
                response = requests.get(url, headers=headers_list[k])
                time.sleep(3)
                IS_USER = response.ok
                if IS_USER:
                    soup = BeautifulSoup(response.text, "html.parser")
                    while "captcha" in soup.text.lower() and k != len(headers_list) - 1:
                        await update.message.reply_text("Получили капчу, пробуем другие headers...")
                        k += 1
                        response = requests.get(f"https://www.kinopoisk.ru/user/{USER_ID}/", headers=headers_list[k])
                        time.sleep(3)
                        soup = BeautifulSoup(response.text, "html.parser")
                    if "captcha" in soup.text.lower():
                        await update.message.reply_text("Опять капча. Попробуйте позже")
                        await go_on(update, context)
                    else:
                        rates_count = soup.find("ul", class_="header_grid_list clearfix").find("b").get_text(strip=True)
                        await update.message.reply_text("Пользователь подтверждён")
                        if rates_count and int(rates_count) >= 30:
                            name = soup.find("div", class_="nick_name").get_text(strip=True)
                            CURRENT_USER = USER_ID
                            CHAT_USERS[CURRENT_USER] = {"state": "default", "name": name, "rates": False,
                                                        "parsed_movies": 0, "api_key": False, "loaded_movies": 0}
                            # users[CURRENT_USER] = CHAT_USERS[CURRENT_USER]
                            # with open("users_database/users.json", "w", encoding="utf-8") as f:
                            #     json.dump(users, f, indent=4, ensure_ascii=False)
                            await update.message.reply_text(f"Здравствуйте, {CHAT_USERS[CURRENT_USER]['name']}")
                            await go_on(update, context)
                        else:
                            await update.message.reply_text("Недостаточно оценок. На профиле должно быть не менее "
                                                            "30 оценок. Оцените больше фильмов или введите другой id")

                else:
                    await update.message.reply_text("Извините, пользователь не найден. "
                                                    "Попробуйте позже или перепроверьте id")
    else:
        await update.message.reply_text("Пожалуйста, введите id пользователя.")


async def get_api(update, context):
    user_input = update.message.text
    if bool(re.fullmatch(r"(([A-Z]|\d){7}-){3}([A-Z]|\d){7}", user_input)):
        with open("auxiliary files/api_keys.txt", "r", encoding="utf-8") as file:
            api_keys = file.read().split("\n")
        if user_input in api_keys:
            CHAT_USERS[CURRENT_USER]["api_key"] = True
        else:
            try:
                headers = {"X-API-KEY": user_input}
                url = 'https://api.kinopoisk.dev/v1.4/movie/373314'
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200 or response.status_code == 403:
                    with open("auxiliary files/api_keys.txt", "a", encoding="utf-8") as file:
                        file.write(user_input + "\n")
                    CHAT_USERS[CURRENT_USER]["api_key"] = True
                    await update.message.reply_text("API-ключ подтверждён")
                else:
                    await update.message.reply_text("Ключ недействителен")
            except requests.exceptions.RequestException:
                await update.message.reply_text("Ошибка проверки ключа. Проблемы с доступом к сети.")
    else:
        await update.message.reply_text("Введённое значение не является ключом")
    await go_on(update, context)


# noinspection PyUnusedLocal
async def add_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
        :param update: Update
        :type context: ContextTypes.DEFAULT_TYPE
        """
    global markup
    global CURRENT_USER
    CURRENT_USER = None
    keyboard = [[InlineKeyboardButton("Назад", callback_data="go_on")],
                [InlineKeyboardButton("Как мне найти id?", callback_data="instructions")]]
    markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Введите id нового пользователя", reply_markup=markup)
    if CURRENT_USER:
        CHAT_USERS[CURRENT_USER]["state"] = "add"


# noinspection PyUnusedLocal
async def change_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
        :param update: Update
        :type context: ContextTypes.DEFAULT_TYPE
        """
    global markup
    if not CHAT_USERS:
        await update.message.reply_text("Пользователей нет. Введите id пользователя",
                                        reply_markup=instructions_markup)
        if CURRENT_USER:
            CHAT_USERS[CURRENT_USER]["state"] = "default"
    else:
        keyboard = []
        for user in CHAT_USERS.keys():
            keyboard.append([InlineKeyboardButton(CHAT_USERS[user]["name"], callback_data=f"user_{user}")])
        keyboard.append([InlineKeyboardButton("Назад", callback_data="go_on")])
        markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Выберите пользователя:", reply_markup=markup)
        if CURRENT_USER:
            CHAT_USERS[CURRENT_USER]["state"] = "user_choice"


async def handle_user_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_USER
    query = update.callback_query
    await query.answer()
    if not CURRENT_USER or CHAT_USERS[CURRENT_USER]["state"] == "user_choice":
        # Получаем ID из callback_data
        data = query.data  # например "user_1"
        user_id = int(data.split("_")[1])

        # Ищем пользователя
        selected = next((CHAT_USERS[u] for u in CHAT_USERS.keys() if u == user_id), None)

        if selected:
            await query.message.reply_text(f"Здравствуйте, {selected['name']}!")
            CURRENT_USER = user_id
            await go_on(update, context)
        else:
            await query.message.reply_text("Пользователь не найден.")
            await go_on(update, context)
    else:
        await query.message.reply_text("Чтобы сменить пользователя введите /change_user")
        await go_on(update, context)
    if CURRENT_USER:
        CHAT_USERS[CURRENT_USER]["state"] = "default"


async def start_over(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global CURRENT_USER
    global CHAT_USERS
    query = update.callback_query
    await query.answer()
    if not CHAT_USERS:
        await query.message.reply_text("Пользователей нет. Введите id пользователя.",
                                       reply_markup=instructions_markup)
    else:
        if not CURRENT_USER or CHAT_USERS[CURRENT_USER]["state"] == "reboot":
            await query.message.reply_text("Чат сброшен. Введите id пользователя",
                                           reply_markup=instructions_markup)
            CHAT_USERS = dict()
            CURRENT_USER = None
        else:
            await query.message.reply_text("Используйте /start для перезапуска чата")
            await go_on(update, context)
        if CURRENT_USER:
            CHAT_USERS[CURRENT_USER]["state"] = "default"


# noinspection PyUnusedLocal
async def go_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global markup
    global CHAT_USERS
    global sent
    global s
    """
        :param update: Update
        :type context: ContextTypes.DEFAULT_TYPE
        """
    chat_id = update.effective_chat.id
    data = context.chat_data
    if update.message:
        update_query = update
    else:
        update_query = update.callback_query

    if CHAT_USERS:
        if CURRENT_USER:
            with open("users_database/users.json", "r", encoding="utf-8") as f:
                users = json.load(f)
            users[str(CURRENT_USER)] = CHAT_USERS[CURRENT_USER]
            with open("users_database/users.json", "w", encoding="utf-8") as f:
                json.dump(users, f)
            CHAT_USERS[CURRENT_USER]["state"] = "default"
            # if not CHAT_USERS[CURRENT_USER]["rates_count"]:
            #     markup = InlineKeyboardMarkup([[InlineKeyboardButton("Начать", callback_data="get_rates_count")]])
            #     await update_query.message.reply_text("Нужно получить количество оценок", reply_markup=markup)
            # else:
            if not CHAT_USERS[CURRENT_USER]["rates"]:
                markup = InlineKeyboardMarkup([[InlineKeyboardButton("Запустить парсинг оценок",
                                                                     callback_data="start_parse")],
                                               [InlineKeyboardButton("Как скачать файлы страниц оценок?",
                                                                     callback_data="file_instructions")]])
                mes = (f"Запустите парсинг оценок либо скиньте файлы страниц оценок. Сейчас ожидается .html файл "
                       f"страницы:\nhttps://www.kinopoisk.r"
                       f"u/user/{CURRENT_USER}/votes/list/vs/vote/perpage/200/page/"
                       f"{CHAT_USERS[CURRENT_USER]['parsed_movies'] // 200 + 1}/#list")
                sent = await update_query.message.reply_text(mes, reply_markup=markup)
            else:
                if not CHAT_USERS[CURRENT_USER]["api_key"]:
                    await update_query.message.reply_text("Получите API-ключ в боте @kinopoiskdev_bot и введите его.")
                    CHAT_USERS[CURRENT_USER]["state"] = "API"
                else:
                    if CHAT_USERS[CURRENT_USER]["loaded_movies"] != CHAT_USERS[CURRENT_USER]["parsed_movies"]:
                        s = await update_query.message.reply_text("Начать загрузку фильмов?",
                                                                  reply_markup=InlineKeyboardMarkup
                                                                  ([[InlineKeyboardButton(
                                                                      "Да",
                                                                      callback_data="start_api_load")]]))
                    else:
                        if not os.path.exists(f"user_models/{CURRENT_USER}.pkl"):
                            await update_query.message.reply_text("Начать подготовку данных?",
                                                                  reply_markup=InlineKeyboardMarkup
                                                                  ([[InlineKeyboardButton(
                                                                      "Да",
                                                                      callback_data="start_data_processing")]]))
                        else:
                            CHAT_USERS[CURRENT_USER]["state"] = "movie_id"
                            await update_query.message.reply_text(
                                "Введите id фильма для предсказания")
        elif CHAT_USERS:
            keyboard = []
            for user in CHAT_USERS.keys():
                keyboard.append([InlineKeyboardButton(CHAT_USERS[user]["name"], callback_data=f"user_{user}")])
            markup = InlineKeyboardMarkup(keyboard)
            await update_query.message.reply_text("Выберите пользователя или введите id", reply_markup=markup)
        else:
            await update_query.message.reply_text("Введите id пользователя.",
                                                  reply_markup=instructions_markup)
    else:
        await update_query.message.reply_text("Введите id пользователя.",
                                              reply_markup=instructions_markup)


def is_empty(soup):
    return "Ни одной записи не найдено" in soup.text


async def parse_page_tg(url, query, headers_index: int):
    try:

        headers = headers_list[headers_index]
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        time.sleep(3)
        if soup and ("robot" in soup.text.lower() or "captcha" in soup.text.lower()):
            await query.message.reply_text("⚠️ Обнаружена капча или блокировка.")
            return "captcha"

        # soup = BeautifulSoup(response.text, 'html.parser')

        if "Ни одной записи не найдено" in soup.text:
            await query.message.reply_text("ℹ️ Пустая страница — ни одной оценки.")
            return []

        name_rus_tags = soup.find_all('div', class_='nameRus')
        vote_tags = soup.find_all('div', class_='vote')[2:]

        if not name_rus_tags:
            await query.message.reply_text("✘ Не удалось найти нужные элементы на странице.")
            return []

        await query.message.reply_text("✓ Контент найден. Парсим...")
        movies = []
        for name_rus, vote in zip(name_rus_tags, vote_tags):
            title_tag = name_rus.find('a')
            if not title_tag:
                continue
            title = title_tag.text.strip()
            link = f"https://www.kinopoisk.ru{title_tag['href']}"
            rating = vote.text.strip()
            if "film" in link:
                movies.append({'title': title, 'link': link, 'rating': rating})
        return movies

    except Exception as e:
        await query.message.reply_text(f"✘ Ошибка при обработке {url}: {e}")
        return "mistake"


def load_parsed_pages(user_id):
    if os.path.exists(f'parsed_pages/{user_id}.txt'):
        with open(f'parsed_pages/{user_id}.txt') as f:
            return set(f.read().splitlines())
    return set()


def save_parsed_page(page_id, user_id):
    with open(f'parsed_pages/{user_id}.txt', 'a') as f:
        f.write(f"{page_id}\n")


def load_existing_movies(user_id):
    if os.path.exists(f'parsed_rates/{user_id}.json'):
        with open(f'parsed_rates/{user_id}.json', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_movies(data, user_id):
    with open(f'parsed_rates/{user_id}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


async def get_rates_tg(update, context):
    if CURRENT_USER and not CHAT_USERS[CURRENT_USER]["rates"]:
        query = update.callback_query
        d_id = CURRENT_USER
        all_movies = load_existing_movies(d_id)
        g_flag = True
        first_page_num = CHAT_USERS[CURRENT_USER]["parsed_movies"] // 200 + 1
        for i in range(first_page_num, 1000):

            url = f"https://www.kinopoisk.ru/user/{d_id}/votes/list/vs/vote/perpage/200/page/{i}/#list"
            j = random.randint(1, 10)
            k = 0
            movies = await parse_page_tg(url, query, j)
            while movies == "captcha" and k < 3:
                await query.message.reply_text("Капча. Пробуем другие header...")
                k += 1
                j = random.randint(1, 10)
                movies = await parse_page_tg(url, query, j)
            if movies == "captcha":
                await query.message.reply_text("Капча. Сбор страниц приостановлен")
                g_flag = False
                break
            elif movies == "mistake":
                await query.message.reply_text("Ошибка сборки.")
                g_flag = False
                break
            elif not movies:
                await query.message.reply_text(f"✓ Конец — пустая страница {i}")
                break

            all_movies += movies
            CHAT_USERS[CURRENT_USER]["parsed_movies"] += len(movies)
            await query.message.reply_text(
                f"+ Страница {i} — добавлено {len(movies)} фильмов (всего: {len(all_movies)})")

            if len(movies) < 50:
                await query.message.reply_text(f"✓ Достигнута последняя неполная страница ({len(movies)} фильмов)")
                break

            time.sleep(5)

        save_movies(all_movies, d_id)
        await query.message.reply_text(f"✔ Готово! Всего фильмов собрано: {len(all_movies)}")
        if g_flag:
            await query.message.reply_text("Все оценки собраны")
            CHAT_USERS[CURRENT_USER]['rates'] = True
        else:
            await query.message.reply_text("Не все оценки собраны")

        await go_on(update, context)
    else:
        await go_on(update, context)


async def go_on_query(update, context):
    query = update.callback_query
    await query.answer()
    await go_on(update, context)


async def show_instructions(update, context):
    query = update.callback_query
    await query.answer()
    with open("images/super-ready.jpg", "rb") as img:
        await query.message.reply_photo(photo=InputFile(img))
    # bot.send_photo(chat_id=update.message.chat_id, photo=img)
    await go_on(update, context)


async def file_instructions(update, context):
    query = update.callback_query
    await query.answer()
    # with open("images/super-ready.jpg", "rb") as img:
    #     await query.message.reply_photo(photo=InputFile(img), caption="Вот как найти свой ID на Кинопоиске 👇")
    # bot.send_photo(chat_id=update.message.chat_id, photo=img)
    if CURRENT_USER:
        mes = (
            f"Чтобы сохранить страницы с оценками, либо используйте скрипт, либо на каждой ожидаемой странице "
            f"используйте Ctrl+S или ПКМ → «Сохранить как», чтобы сохранить её на компьютер. Полученные "
            f"файлы прикрепите к сообщению и отправьте."
        )
        await query.message.reply_text(mes, reply_markup=script_markup)
    else:
        await update.message.reply_text("Текущий пользователь отсутствует.")
    await go_on(update, context)


async def show_script(update, context):
    query = update.callback_query
    await query.answer()
    # with open("images/super-ready.jpg", "rb") as img:
    #     await query.message.reply_photo(photo=InputFile(img), caption="Вот как найти свой ID на Кинопоиске 👇")
    # bot.send_photo(chat_id=update.message.chat_id, photo=img)
    kp_script = f"""
    (async () => {{
      const userId = "{CURRENT_USER}";
      const baseUrl = `https://www.kinopoisk.ru/user/${{userId}}/votes/list/vs/vote/perpage/200/page`;
      const downloadPage = async (i) => {{
        const url = `${{baseUrl}}/${{i}}/#list`;
        const res = await fetch(url, {{ credentials: 'include' }});
        const text = await res.text();

        if (text.includes("Ни одной записи не найдено")) {{
          console.log(`🛑 Страница ${{i}} пуста. Останавливаемся.`);
          return null;
        }}

        const blob = new Blob([text], {{ type: 'text/html' }});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `kp_page_${{i}}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        console.log(`✅ Скачано: страница ${{i}}`);
        return true;
      }};

      let i = 1;
      while (true) {{
        const ok = await downloadPage(i);
        if (!ok) break;
        i++;
        await new Promise(r => setTimeout(r, 1200));
      }}

      console.log("📦 Все доступные страницы скачаны.");
    }})();
    """
    await query.message.reply_text(kp_script, reply_markup=script_instructions_markup)

    await go_on(update, context)


async def show_script_instructions(update, context):
    query = update.callback_query
    await query.answer()
    await query.message.reply_text(
        "Для использования скрипта:\n"
        "1. Откройте страницу: https://www.kinopoisk.ru/\n"
        "2. Нажмите F12, перейдите во вкладку 'Console'\n"
        "3. Вставьте скрипт. Если скрипт не вставляется, введите команду: `allow paste`\n"
        "4. Если браузер попросит разрешение на скачивание нескольких файлов — разрешите\n"
        "📁 Файлы будут сохранены в папку 'Загрузки'"
    )
    await go_on(update, context)


async def handle_document(update, context):
    global sent
    if sent:
        await context.bot.delete_message(chat_id=sent.chat.id, message_id=sent.message_id)
    file = await context.bot.get_file(update.message.document.file_id)
    buffer = BytesIO()
    await file.download_to_memory(out=buffer)
    if not CURRENT_USER or CHAT_USERS[CURRENT_USER]["rates"]:
        await update.message.reply_text("Файл не ожидается")

    try:
        html_text = buffer.getvalue().decode("utf-8")
        soup = BeautifulSoup(html_text, 'html.parser')
        user_id = soup.find("p",
                            class_="profile_name profile_name_link_box js-rum-hero").find("a")['href'].split("/")[-2]
        if CURRENT_USER != int(user_id):
            await update.message.reply_text("Файл оценок не соответствует текущему пользователю")
        else:
            from_to = soup.find("div", class_="pagesFromTo").text
            from_ = int(from_to.split("—")[0])
            to_ = int(from_to.split("—")[1].split()[0])
            of_ = int(from_to.split("—")[1].split()[-1])

            if (to_ - from_ == 199 or to_ == of_) and CHAT_USERS[CURRENT_USER]["parsed_movies"] == from_ - 1:
                all_movies = load_existing_movies(CURRENT_USER)
                CHAT_USERS[CURRENT_USER]["parsed_movies"] += to_ - from_ + 1
                await update.message.reply_text(f"Собраны {CHAT_USERS[CURRENT_USER]['parsed_movies']} из {of_} файлов")
                if CHAT_USERS[CURRENT_USER]["parsed_movies"] == of_:
                    CHAT_USERS[CURRENT_USER]["rates"] = of_
                    await update.message.reply_text("Все оценки собраны!")
                name_rus_tags = soup.find_all('div', class_='nameRus')
                vote_tags = soup.find_all('div', class_='vote')[2:]

                if not vote_tags:
                    vote_tags = re.findall(r"rating: '(\d+)'", str(soup))
                else:
                    vote_tags = [vote.text.strip() for vote in vote_tags]
                movies = []
                for name_rus, vote in zip(name_rus_tags, vote_tags):
                    title_tag = name_rus.find('a')
                    if not title_tag:
                        continue
                    title = title_tag.text.strip()
                    link = f"https://www.kinopoisk.ru{title_tag['href']}"
                    rating = vote
                    if "film" in link:
                        movies.append({'title': title, 'link': link, 'rating': rating})
                all_movies += movies
                save_movies(all_movies, CURRENT_USER)
            else:
                await update.message.reply_text(f"Ожидается .html-файл страницы 'https://www.kinopoisk.ru/user/"
                                                f"{CURRENT_USER}/votes/list/vs/vote/perpage/200/page/"
                                                f"{CHAT_USERS[CURRENT_USER]['parsed_movies'] // 200 + 1}/#list' "
                                                f"(`kp_page_{CHAT_USERS[CURRENT_USER]['parsed_movies'] // 200 + 1}"
                                                f".html`)")
    except (AttributeError, TypeError, ValueError, IndexError):
        await update.message.reply_text("Неверный файл. Убедитесь, что вы сохранили нужную страницу целиком.")
    await go_on(update, context)


async def start_data_processing(update, context):
    query = update.callback_query
    await query.answer()
    if (CURRENT_USER and CHAT_USERS[CURRENT_USER]["loaded_movies"] and CHAT_USERS[CURRENT_USER]["loaded_movies"] ==
            CHAT_USERS[CURRENT_USER]["parsed_movies"] and not os.path.exists(f"user_models/{CURRENT_USER}.pkl")):
        await query.message.reply_text("Подготавливаем данные и сохраняем модель...")
        train_and_ensemble(f"raw_movie_files/{CURRENT_USER}.json", f"user_models/{CURRENT_USER}.pkl")

    await go_on(update, context)


async def start_api_load(update, context):
    global s
    query = update.callback_query
    await query.answer()
    if CURRENT_USER and CHAT_USERS[CURRENT_USER]["api_key"] and CHAT_USERS[CURRENT_USER]["loaded_movies"] != \
            CHAT_USERS[CURRENT_USER]["parsed_movies"]:
        s = await query.message.reply_text("Загружаем фильмы...")
        url = f"parsed_rates/{CURRENT_USER}.json"
        with open(url, "r", encoding="utf-8") as f:
            all_rates = json.load(f)
        movie_ids = [movie["link"].split("/")[-2] for movie in all_rates]
        ratings = [movie["rating"] for movie in all_rates]
        ratings_dict = dict()
        for i in range(len(movie_ids)):
            ratings_dict[movie_ids[i]] = ratings[i]

        with open("auxiliary files/api_keys.txt", "r", encoding="utf-8") as f:
            api_keys = f.read().split("\n")
        random.shuffle(api_keys)
        BASE_URL = 'https://api.kinopoisk.dev/v1.4/movie/'
        if os.path.exists(f"raw_movie_files/{CURRENT_USER}.json"):
            with open(f"raw_movie_files/{CURRENT_USER}.json", encoding='utf-8') as f:
                movies_data = json.load(f)
            existing_ids = {str(movie.get("id")) for movie in movies_data if "id" in movie}
        else:
            movies_data = []
            existing_ids = set()
        key_index = 0
        requests_left = 200
        headers = {"X-API-KEY": api_keys[key_index]}

        def switch_key():
            nonlocal key_index, headers, requests_left
            key_index += 1
            if key_index >= len(api_keys):
                return False
            headers = {"X-API-KEY": api_keys[key_index]}
            requests_left = 200
            return True

        def get_movie_data(m_id):
            nonlocal requests_left
            base_url = f"{BASE_URL}{m_id}"
            for attempt in range(3):
                try:
                    response = requests.get(base_url, headers=headers, timeout=10)
                    if response.status_code == 403:
                        return None
                    if response.status_code == 429:
                        return None
                    if response.status_code != 200:
                        return None
                    if response.text == "LIMIT":
                        return None
                    return response.json()
                except requests.exceptions.RequestException:
                    time.sleep(5)
            return None

        flag = True
        result = True
        for movie_id in movie_ids:
            if movie_id in existing_ids:
                continue
            while flag:
                if requests_left <= 0:
                    if not switch_key():
                        await query.message.reply_text(f"Ключи исчерпаны. Попробуйте завтра.")
                        flag = False
                        break

                result = get_movie_data(movie_id)
                if result is None:
                    if not switch_key():
                        await query.message.reply_text(f"Ключи исчерпаны. Попробуйте завтра.")
                        flag = False
                        break
                    continue  # пробуем снова с новым ключом
                break  # вышли из while, получили результат
            if isinstance(result, dict) and result:
                result["MYrating"] = int(ratings_dict[movie_id])
                movies_data.append(result)
                CHAT_USERS[CURRENT_USER]["loaded_movies"] = len(movies_data)
                with open(f"raw_movie_files/{CURRENT_USER}.json", 'w', encoding='utf-8') as f:
                    json.dump(movies_data, f, ensure_ascii=False, indent=2)
                existing_ids.add(movie_id)
                requests_left -= 1
                if s:
                    await context.bot.edit_message_text(
                        text=f"Загружено {len(movies_data)} из {CHAT_USERS[CURRENT_USER]['parsed_movies']}",
                        chat_id=s.chat.id,
                        message_id=s.message_id
                    )

        if flag:
            await query.message.reply_text("Все фильмы загружены")
    # else:
    #     await query.message.reply_text(f"Загружено {CHAT_USERS[CURRENT_USER]['loaded_movies']} из "
    #                                    f"{CHAT_USERS[CURRENT_USER]['parsed_movies']} фильмов")
    await go_on(update, context)


async def print_prediction(update, context):
    with open("auxiliary files/api_keys.txt", "r", encoding="utf-8") as f:
        api_keys = f.read().split("\n")
    BASE_URL = 'https://api.kinopoisk.dev/v1.4/movie/'
    k = 0
    headers = {"X-API-KEY": api_keys[k]}
    movie_id = update.message.text
    base_url = f"{BASE_URL}{movie_id}"
    response = requests.get(base_url, headers=headers)
    while response.status_code == 403 and k != len(api_keys) - 1:
        k += 1
        headers = {"X-API-KEY": api_keys[k]}
        response = requests.get(base_url, headers=headers)
    if response.status_code == 403:
        await update.message.reply_text("API-ключи исчерпаны. Попробуйте позже.")
    elif not response.ok:
        await update.message.reply_text("ID введён неверно.")
    else:
        await update.message.reply_text("Предсказываем...")
        result = response.json()
        with open(f"user_models/{CURRENT_USER}.pkl", 'rb') as f:
            loaded_model = pickle.load(f)
        X_new = pd.DataFrame([result])
        predictions = loaded_model.predict(X_new)
        results_df = pd.DataFrame({
            'id': X_new.get('id', 'N/A'),
            'name': X_new.get('name', 'N/A'),
            'predicted_rating': predictions
        })
        results_df['predicted_rating'] = np.clip(results_df['predicted_rating'], 1, 10)
        movie_name = X_new.iloc[0].get('name', 'Без названия')
        predicted_rating = predictions[0]

        # Ограничиваем и форматируем рейтинг
        rating_str = f"{predicted_rating:.2f}"

        # Собираем простое текстовое сообщение без Markdown
        final_message = (
            f"✨ Прогноз для фильма «{movie_name}»:\n\n"
            f"🎬 Ваш предполагаемый рейтинг: {rating_str} / 10"
        )

        # В коде вашего бота эта строка теперь будет выглядеть так:
        # Она отправит сообщение как есть, без попытки что-либо отформатировать.
        await update.message.reply_text(final_message)
    await go_on(update, context)

load_dotenv()
BOT_TOKEN = os.environ.get("BOT_TOKEN")
API_KEYS = os.environ.get("API_KEYS")
API_KEYS_LIST = API_KEYS.split(" ")
API_S = "\n".join(API_KEYS_LIST)
with open("auxiliary files/api_keys.txt", "w", encoding="utf-8") as fi:
    fi.write(API_S)
headers_list = [{
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "Referer": "https://www.kinopoisk.ru/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
},
                   {
                       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/134.0.0.0"
                                     "YaBrowser/25.4.0.0 Safari/537.36",
                       "Accept": "*/*",
                       "Accept-Language": "ru,en;q=0.9",
                       "Referer": "https://www.kinopoisk.ru/",
                       # "Cookie": ,
                       "Cache-Control": "no-cache",
                       "Connection": "keep-alive",
                       "Sec-Fetch-Dest": "empty",
                       "Sec-Fetch-Mode": "no-cors",
                       "Sec-Fetch-Site": "cross-site",
                       "sec-fetch-storage-access": "active"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/134.0.0.0"
                                     "YaBrowser/25.4.0.0 Safari/537.36",
                       "Accept": "*/*",
                       "Accept-Language": "ru,en;q=0.9",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Cookie": "yandexuid=8969119991720263528; yashr=926658521720263535; yuidss=8969119991720263528; ymex=2035623597.yrts.1720263597; yandex_login=egor.matiuha; my=YwA=; gdpr=0; _ym_uid=1688552967950758693; amcuid=4355776901720429538; bh=EjsiTm90L0EpQnJhbmQiO3Y9IjgiLCAiQ2hyb21pdW0iO3Y9IjEyNiIsICJZYUJyb3dzZXIiO3Y9IjI0IhoFIng4NiIiDCIyNC42LjIuNzcwIioCPzA6CSJXaW5kb3dzIkIIIjE1LjAuMCJKBCI2NCJSVCJDaHJvbWl1bSI7dj0iMTI0LjAuNjM2Ny4yNDMiLCJZYUJyb3dzZXIiO3Y9IjI0LjYuMi43NzAiLCJOb3QtQS5CcmFuZCI7dj0iOTkuMC4wLjAiIg==; _ym_d=1742546859; yabs-vdrf=A0; is_gdpr=0; is_gdpr_b=CK6UEBDhwwIoAg==; i=GanrXs1XzY58D0xpW0okLnMmD88wyCyMNc6ubDosaRotbXYCZalN1vyLaMAJOItonahfUZwhrIXjW8EuOJn3B7n17I4=; Session_id=3:1749558614.5.0.1720263601806:LRgVJQ:13.1.1:czoxNzE3OTMyMTI3NDUzOl9oTVExQToyNQ.2:1|931479797.26651430.2.2:26651430.3:1746915031|3:10308765.819479.n3V9_jOhMqhNfTUmckHRd68lEEw; sessar=1.1202.CiAS11C_CzQ9dY386XEkGIGEyP5J6rxP9KhG-7z1KCxkBw.W_v9xbONalZrMTT3-vOEp6szk8AUE6-2DXddwBy4iXw; sessionid2=3:1749558614.5.0.1720263601806:LRgVJQ:13.1.1:czoxNzE3OTMyMTI3NDUzOl9oTVExQToyNQ.2:1|931479797.26651430.2.2:26651430.3:1746915031|3:10308765.819479.fakesign0000000000000000000; cycada=IZHMfCHwUQWWFHAzgzM5ggqMea+WoxkAfaiXat73ZhQ=; isa=joMufcIu11wIwtU2SgfODznjTuasP3ZFK9orYVUkjH/sv3Xz8CCKCUtlGQ1QTsjtKW9GV1gfg7F1yofqWHuiyVSHA90=; sae=0:8EFAB2AD-308F-423A-A140-37D2D47D0883:b:25.4.4.530:w:d:RU:20230705; yp=1749820701.uc.ru#1749820701.duc.ru#1758806079.brd.6400000000#1758806079.cld.2270452#1752236470.csc.1#1779455220.dafs.5-3_6-3_7-3_10-3#1750886647.hdrc.1#2057912811.hks.0#2065021168.pcs.1#4294967295.skin.l#1781197168.swntab.0#1764651761.szm.1_25:1536x864:1519x740#1772800326.dc_neuro.10#1772799882.bk-map.1#2062275031.udn.cDrQldCz0L7RgCDQnC4%3D#1750525168.dlp.1; ys=udn.cDrQldCz0L7RgCDQnC4%3D#c_chck.3606030996; bh=ElAiQ2hyb21pdW0iO3Y9IjEzNCIsICJOb3Q6QS1CcmFuZCI7dj0iMjQiLCAiWWFCcm93c2VyIjt2PSIyNS40IiwgIllvd3NlciI7dj0iMi41IhoFIng4NiIiDCIyNC42LjIuNzcwIioCPzA6CSJXaW5kb3dzIkIIIjE1LjAuMCJKBCI2NCJSZyJDaHJvbWl1bSI7dj0iMTM0LjAuNjk5OC41MzAiLCAiTm90OkEtQnJhbmQiO3Y9IjI0LjAuMC4wIiwgIllhQnJvd3NlciI7dj0iMjUuNC40LjUzMCIsICJZb3dzZXIiO3Y9IjIuNSJaAj8wYManq8IGaiHcyuH/CJLYobEDn8/h6gP7+vDnDev//fYPx4OClwbzgQI=",
                       "Cache-Control": "no-cache",
                       "Connection": "keep-alive",
                       "Sec-Fetch-Dest": "empty",
                       "Sec-Fetch-Mode": "no-cors",
                       "Sec-Fetch-Site": "cross-site",
                       "sec-fetch-storage-access": "active"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
                       "Accept": "*/*",
                       "Accept-Language": "ru,en;q=0.9,en-GB;q=0.8,en-US;q=0.7",
                       "Referer": "https://www.kinopoisk.ru/",
                       # "Cookie": ,
                       "Sec-Fetch-Dest": "empty",
                       "Sec-Fetch-Mode": "cors",
                       "Sec-Fetch-Site": "cross-site",
                       "sec-fetch-storage-access": "active"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/137.0.0.0 Safari/537.36 Edg/137.0.0.0",
                       "Accept": "*/*",
                       "Accept-Language": "ru,en;q=0.9,en-GB;q=0.8,en-US;q=0.7",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Cookie": "yandexuid=2828172591688552891; yuidss=2828172591688552891; ymex=2044858868.yrts.1729498868; yashr=2280337591733872876; _ym_uid=1688552916983330593; receive-cookie-deprecation=1; yandex_login=egor.matiuha; gdpr=0; amcuid=8805531571737317651; my=YwA=; is_gdpr=0; is_gdpr_b=CKKpXBCXtwIoAg==; yabs-vdrf=CcSvdI00_bSS1KuvdTW2Z0W000; _ym_d=1745602712; yp=1756689932.szm.1_25%3A1536x864%3A1513x738%3A15#2062754570.pcs.1#1778930882.swntab.0#1748683452.hdrc.0#1777138754.dc_neuro.2#1748258570.dlp.2; i=6OsqtVXrO1iRqA9SkRJME/C9uSJrUkqKlvou/+ZJZRsMBVfLvje8EcBwcnTfqfMYwcxh5YBoiG7z6owG3pYbizNfHiE=; Session_id=3:1749810342.5.0.1735330185726:Ep3mJQ:e552.1.2:1|931479797.0.2.3:1735330185|3:10308905.408328._4XdoXCFWtGMaJfbQSi0CcfpltU; sessar=1.1202.CiBFuwmLI2zrrlVvq8BhPZhWU3zBHme6v_r0SejCY3uEUA.OYqOmH8EafawqxjMCSas6hky_uaFGIeaKMzsTfl3IbI; sessionid2=3:1749810342.5.0.1735330185726:Ep3mJQ:e552.1.2:1|931479797.0.2.3:1735330185|3:10308905.408328.fakesign0000000000000000000; ys=udn.cDrQldCz0L7RgCDQnC4%3D#c_chck.3632214097; bh=EkIiTWljcm9zb2Z0IEVkZ2UiO3Y9IjEzNyIsICJDaHJvbWl1bSI7dj0iMTM3IiwgIk5vdC9BKUJyYW5kIjt2PSIyNCIaBSJ4ODYiIg8iMTIwLjAuMjIxMC42MSIqAj8wOgkiV2luZG93cyJCCCIxNS4wLjAiSgQiNjQiUloiTm90X0EgQnJhbmQiO3Y9IjguMC4wLjAiLCJDaHJvbWl1bSI7dj0iMTIwLjAuNjA5OS43MSIsIk1pY3Jvc29mdCBFZGdlIjt2PSIxMjAuMC4yMjEwLjYxIiJgyvmvwgZqIdzK4f8IktihsQOfz+HqA/v68OcN6//99g/C88yHCOOHAg==",
                       "Sec-Fetch-Dest": "empty",
                       "Sec-Fetch-Mode": "cors",
                       "Sec-Fetch-Site": "cross-site",
                       "sec-fetch-storage-access": "active"
                   },
                   {
                       'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                     "Chrome/124.0.6367.770 YaBrowser/24.6.2.770 (beta) Yowser/2.5 Safari/537.36"

                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                       "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive",
                       "Upgrade-Insecure-Requests": "1"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                       "Accept-Language": "ru,en-US;q=0.7,en;q=0.3",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 "
                                     "(KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                       "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) YaBrowser/24.3.0.0 Yowser/2.5 Chrome/"
                                     "124.0.6367.90 Safari/537.36",
                       "Accept": "*/*",
                       "Accept-Language": "ru,en;q=0.9",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                       "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) Chrome/124.0.6367.78 Safari/537.36",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                       "Accept-Language": "ru,en;q=0.8",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_4 like Mac OS X) "
                                     "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                       "Accept-Language": "ru-RU,ru;q=0.9",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Android 13; Mobile; rv:123.0) Gecko/123.0 Firefox/123.0",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                       "Accept-Language": "ru-RU,ru;q=0.9",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (iPad; CPU OS 15_5 like Mac OS X) AppleWebKit/605.1.15 "
                                     "(KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                       "Accept-Language": "ru,en;q=0.9",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive"
                   },
                   {
                       "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) Chrome/120.0.6099.110 Safari/537.36",
                       "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                       "Accept-Language": "ru,en-US;q=0.8",
                       "Referer": "https://www.kinopoisk.ru/",
                       "Connection": "keep-alive"
                   }
               ][::-1]
instructions_markup = InlineKeyboardMarkup([[InlineKeyboardButton("Где мне найти свой id?",
                                                                  callback_data="show_instructions")]])
script_markup = InlineKeyboardMarkup([[InlineKeyboardButton("Показать скрипт",
                                                            callback_data="show_script")]])
script_instructions_markup = InlineKeyboardMarkup([[InlineKeyboardButton("Как использовать скрипт?",
                                                                         callback_data="script_instructions")]])
sent: Optional[Message] = None
s: Optional[Message] = None

markup = InlineKeyboardMarkup(list())
CURRENT_USER = None
CHAT_USERS = dict()

app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("add_user", add_user))
app.add_handler(CommandHandler("change_user", change_user))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
app.add_handler(CallbackQueryHandler(handle_user_choice, pattern=r"^user_\d+$"))
app.add_handler(CallbackQueryHandler(start_over, pattern="start_over"))
app.add_handler(CallbackQueryHandler(go_on_query, pattern="go_on"))
app.add_handler(CallbackQueryHandler(get_rates_tg, pattern="start_parse"))
app.add_handler(CallbackQueryHandler(show_instructions, pattern="show_instructions"))
app.add_handler(CallbackQueryHandler(file_instructions, pattern="file_instructions"))
app.add_handler(CallbackQueryHandler(show_script, pattern="show_script"))
app.add_handler(CallbackQueryHandler(show_script_instructions, pattern="script_instructions"))
app.add_handler(CallbackQueryHandler(start_api_load, pattern="start_api_load"))
app.add_handler(CallbackQueryHandler(start_data_processing, pattern="start_data_processing"))

app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

app.run_webhook()
