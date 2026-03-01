import pandas as pd


def prepare_dataset(raw_csv_path: str, output_csv_path: str) -> None:
    print(f"📥 Завантаження сирих даних з {raw_csv_path}...")

    # 1. Читаємо CSV
    df = pd.read_csv(raw_csv_path)

    # 2. Перетворюємо колонку 'ts' у формат datetime
    df["ts"] = pd.to_datetime(df["ts"])

    # 3. Округлюємо час до найближчих 15 секунд.
    # Це згрупує CPU, RAM та RPS, які Worker зібрав в одному циклі, під одним і тим самим таймстемпом.
    df["ts_rounded"] = df["ts"].dt.round("15s")

    # 4. Робимо "Pivot" (зведену таблицю).
    # Беремо саме input_value, бо це те реальне значення, яке Worker надсилав у момент часу ts.
    print("🔄 Трансформація формату (Long -> Wide)...")
    clean_df = df.pivot_table(
        index="ts_rounded",
        columns="resource",
        values="input_value",
        aggfunc="mean",  # Якщо випадково попадуть два CPU в одну секунду - візьмемо середнє
    )

    # 5. Наводимо порядок у колонках (щоб порядок фіч завжди був однаковий для LSTM)
    clean_df = clean_df[["cpu", "ram", "rps"]]

    # 6. Заповнюємо пропуски (Forward Fill).
    # Якщо метрика RPS не зібралася в якісь 15 секунд, беремо значення з попереднього кроку.
    missing_before = clean_df.isnull().sum().sum()
    clean_df = clean_df.ffill().bfill()
    missing_after = clean_df.isnull().sum().sum()

    if missing_before > 0:
        print(f"🩹 Відновлено {missing_before} пропущених значень.")

    # 7. Зберігаємо ідеальний датасет
    clean_df.to_csv(output_csv_path)

    print(
        f"✅ Готово! Збережено {len(clean_df)} рядків (таймстепів) у {output_csv_path}."
    )
    print("\nПерші 5 рядків готового датасету:")
    print(clean_df.head())


if __name__ == "__main__":
    # Назви файлів
    RAW_FILE = "./raw_data/raw_metrics.csv"  # Файл, який ти скачав з pgAdmin
    CLEAN_FILE = "./clean_data/training_data.csv"  # Файл, який ми віддамо нейромережі

    prepare_dataset(RAW_FILE, CLEAN_FILE)
