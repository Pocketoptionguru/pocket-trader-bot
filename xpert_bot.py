from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters,
    CallbackQueryHandler, ConversationHandler, ContextTypes
)

# Conversation states
LANGUAGE, EMAIL, PLAN_SELECTION, PAYMENT_METHOD, CUSTOM_PAYMENT = range(5)

# Admin and payment details
ADMIN_ID = 6361476170
BINANCE_ID = "759931884"
BYBIT_ID = "100141071"
API_TOKEN = "8145954126:AAEByDNj6An7q6BjQ9rwEG9NESkM6pDVL0k"

LANGUAGES = {
    "English": "Welcome to Xpert Trading Bot! Please enter your Gmail to continue.",
    "Hindi": "Xpert Trading Bot में आपका स्वागत है! कृपया जारी रखने के लिए अपना Gmail दर्ज करें।",
    "French": "Bienvenue sur Xpert Trading Bot ! Veuillez entrer votre Gmail pour continuer.",
    "Arabic": "مرحبًا بك في Xpert Trading Bot! الرجاء إدخال بريدك الإلكتروني في Gmail للمتابعة.",
    "German": "Willkommen beim Xpert Trading Bot! Bitte geben Sie Ihre Gmail-Adresse ein, um fortzufahren."
}

PLANS = {
    "Pro Plan": {
        "price": "$75/year",
        "features": "High win rate (95%)\n24/7 support\nTrading signals\n$500/week potential profit"
    },
    "Basic Half-Year Plan": {
        "price": "$45/6 months",
        "features": "Trading entries\nAnalytics\n$500/month potential on $10 deposit"
    }
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(lang, callback_data=lang)] for lang in LANGUAGES]
    await update.message.reply_text("Please select your language:", reply_markup=InlineKeyboardMarkup(keyboard))
    return LANGUAGE


async def language_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = update.callback_query.data
    context.user_data["language"] = lang
    await update.callback_query.answer()
    await update.callback_query.message.reply_text(LANGUAGES[lang])
    return EMAIL


async def collect_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    email = update.message.text
    if "@gmail.com" not in email:
        await update.message.reply_text("Please enter a valid Gmail address ending with @gmail.com.")
        return EMAIL

    context.user_data["email"] = email
    keyboard = [[InlineKeyboardButton(plan, callback_data=plan)] for plan in PLANS]
    await update.message.reply_text("Choose your plan:", reply_markup=InlineKeyboardMarkup(keyboard))
    return PLAN_SELECTION


async def plan_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    plan = update.callback_query.data
    context.user_data["plan"] = plan
    await update.callback_query.answer()

    plan_details = f"{plan}\n{PLANS[plan]['price']}\n\nFeatures:\n{PLANS[plan]['features']}"
    await update.callback_query.message.reply_text(plan_details)

    keyboard = [
        [InlineKeyboardButton("Binance", callback_data="Binance")],
        [InlineKeyboardButton("Bybit", callback_data="Bybit")],
        [InlineKeyboardButton("Other", callback_data="Other")]
    ]
    await update.callback_query.message.reply_text("Select your preferred payment method:", reply_markup=InlineKeyboardMarkup(keyboard))
    return PAYMENT_METHOD


async def payment_method(update: Update, context: ContextTypes.DEFAULT_TYPE):
    method = update.callback_query.data
    context.user_data["payment_method"] = method
    await update.callback_query.answer()

    user_info = context.user_data

    if method == "Binance":
        await update.callback_query.message.reply_text(
            f"Please make payment to this Binance ID:\n\n`{BINANCE_ID}`\n\nSend screenshot after payment.",
            parse_mode="Markdown"
        )
    elif method == "Bybit":
        await update.callback_query.message.reply_text(
            f"Please make payment to this Bybit ID:\n\n`{BYBIT_ID}`\n\nSend screenshot after payment.",
            parse_mode="Markdown"
        )
    else:
        await update.callback_query.message.reply_text("Please type your preferred payment method:")
        return CUSTOM_PAYMENT

    msg = (
        f"NEW USER SIGNUP\n\n"
        f"Language: {user_info.get('language')}\n"
        f"Gmail: {user_info.get('email')}\n"
        f"Plan: {user_info.get('plan')}\n"
        f"Payment Method: {method}"
    )
    await context.bot.send_message(chat_id=ADMIN_ID, text=msg)
    return ConversationHandler.END


async def custom_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    method = update.message.text
    context.user_data["payment_method"] = method
    await update.message.reply_text(f"Thanks. We’ll reach out to verify your custom payment method: {method}")

    user_info = context.user_data
    msg = (
        f"NEW USER SIGNUP\n\n"
        f"Language: {user_info.get('language')}\n"
        f"Gmail: {user_info.get('email')}\n"
        f"Plan: {user_info.get('plan')}\n"
        f"Custom Payment Method: {method}"
    )
    await context.bot.send_message(chat_id=ADMIN_ID, text=msg)
    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Process cancelled.")
    return ConversationHandler.END


if __name__ == "__main__":
    app = ApplicationBuilder().token(API_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            LANGUAGE: [CallbackQueryHandler(language_selection)],
            EMAIL: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_email)],
            PLAN_SELECTION: [CallbackQueryHandler(plan_selection)],
            PAYMENT_METHOD: [CallbackQueryHandler(payment_method)],
            CUSTOM_PAYMENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, custom_payment)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(conv_handler)

    print("Bot is running...")
    app.run_polling()