from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker


NUM_RECORDS = 10_000
OUTPUT_PATH = Path("data/raw/loan_applications.csv")
SEED = 42

fake = Faker("en_US")
Faker.seed(SEED)
random.seed(SEED)


LOAN_PURPOSES = [
    "debt_consolidation",
    "home_improvement",
    "medical",
    "small_business",
    "auto",
    "moving",
    "major_purchase",
    "vacation",
    "wedding",
    "education",
]

EMPLOYERS = [
    "North Ridge Logistics",
    "Blue Harbor Health",
    "Peakstone Retail",
    "Summit Financial Group",
    "Redwood Manufacturing",
    "Crescent Energy",
    "BrightPath Education",
    "UrbanCore Services",
    "Pioneer Foods",
    "Greenline Telecom",
    "Atlas Construction",
    "Silver Oak Pharmacy",
]


def random_date_of_birth(min_age: int = 21, max_age: int = 74) -> date:
    today = date.today()
    earliest = today - timedelta(days=max_age * 365)
    latest = today - timedelta(days=min_age * 365)
    return earliest + timedelta(days=random.randint(0, (latest - earliest).days))


def generate_ssn_for_birth_year(birth_year: int) -> str:
    # Educational simplification:
    # We encode the first three digits to loosely track a birth-decade bucket
    # so we can later inject records where the encoded range conflicts with DOB.
    decade = (birth_year // 10) % 10
    area = 100 + decade * 70 + random.randint(0, 69)
    group = random.randint(10, 99)
    serial = random.randint(1000, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"


def ssn_matches_birth_year(ssn: str, birth_year: int) -> bool:
    area = int(ssn.split("-")[0])
    encoded_decade = (area - 100) // 70
    actual_decade = (birth_year // 10) % 10
    return encoded_decade == actual_decade


def annual_income_for_profile(credit_score: int, time_at_employer_months: int) -> int:
    baseline = random.randint(28_000, 140_000)
    experience_boost = min(time_at_employer_months, 180) * random.randint(60, 120)

    if credit_score >= 760:
        baseline += random.randint(25_000, 70_000)
    elif credit_score >= 680:
        baseline += random.randint(8_000, 30_000)
    elif credit_score <= 580:
        baseline -= random.randint(4_000, 12_000)

    income = baseline + experience_boost
    return max(18_000, min(income, 240_000))


def credit_score_for_profile() -> int:
    roll = random.random()
    if roll < 0.10:
        return random.randint(520, 579)
    if roll < 0.30:
        return random.randint(580, 649)
    if roll < 0.70:
        return random.randint(650, 739)
    if roll < 0.92:
        return random.randint(740, 799)
    return random.randint(800, 850)


def realistic_loan_amount(annual_income: int) -> int:
    ratio = random.uniform(0.08, 0.55)
    amount = int(round(annual_income * ratio / 500.0) * 500)
    return max(2_000, min(amount, 75_000))


def realistic_inquiries(credit_score: int) -> int:
    if credit_score >= 740:
        return random.randint(0, 3)
    if credit_score >= 650:
        return random.randint(1, 5)
    return random.randint(2, 9)


def make_phone_number() -> str:
    return fake.numerify(text="###-###-####")


def build_legit_record(applicant_index: int) -> dict[str, object]:
    full_name = fake.name()
    dob = random_date_of_birth()
    credit_score = credit_score_for_profile()
    time_at_address_months = random.randint(6, 240)
    time_at_employer_months = random.randint(6, 240)
    income = annual_income_for_profile(credit_score, time_at_employer_months)
    loan_amount = realistic_loan_amount(income)

    return {
        "applicant_id": f"APP-{applicant_index:06d}",
        "full_name": full_name,
        "ssn": generate_ssn_for_birth_year(dob.year),
        "date_of_birth": dob.isoformat(),
        "address": fake.address().replace("\n", ", "),
        "employer_name": random.choice(EMPLOYERS),
        "annual_income": income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "loan_purpose": random.choice(LOAN_PURPOSES),
        "email": fake.email(),
        "phone_number": make_phone_number(),
        "time_at_address_months": time_at_address_months,
        "time_at_employer_months": time_at_employer_months,
        "num_recent_inquiries": realistic_inquiries(credit_score),
        "is_fraud": 0,
    }


def inject_ssn_birth_year_mismatch(record: dict[str, object]) -> None:
    # Fraud signal 1:
    # The synthetic SSN format encodes a rough birth-decade bucket in the
    # first three digits. Fraudsters using fabricated identities may pair an
    # SSN pattern with a date of birth that falls outside the expected range.
    dob_year = int(str(record["date_of_birth"])[:4])
    mismatched_year = dob_year
    while mismatched_year % 10 == dob_year % 10:
        mismatched_year = random.choice([1962, 1977, 1984, 1991, 2006])
    record["ssn"] = generate_ssn_for_birth_year(mismatched_year)
    record["is_fraud"] = 1


def inject_unrealistic_income_to_loan_ratio(record: dict[str, object]) -> None:
    # Fraud signal 2:
    # A loan request that is far too large relative to verified income can be
    # a sign of first-party fraud or a synthetic identity attempting to
    # maximize proceeds before default.
    income = int(record["annual_income"])
    multiplier = random.uniform(1.7, 3.8)
    record["loan_amount"] = int(round(income * multiplier / 500.0) * 500)
    record["is_fraud"] = 1


def inject_short_stability(record: dict[str, object]) -> None:
    # Fraud signal 3:
    # Extremely short tenure at both the current address and current employer
    # indicates instability that becomes more suspicious when both happen
    # simultaneously instead of independently.
    record["time_at_address_months"] = random.randint(0, 3)
    record["time_at_employer_months"] = random.randint(0, 3)
    record["is_fraud"] = 1


def inject_duplicate_ssn(records: list[dict[str, object]], rng: random.Random) -> None:
    # Fraud signal 4:
    # Re-using the same SSN across multiple applications while changing the
    # applicant name is a classic synthetic or identity-theft indicator.
    duplicate_groups = max(80, len(records) // 90)
    source_indices = rng.sample(range(len(records)), duplicate_groups)

    for source_idx in source_indices:
        source_record = records[source_idx]
        clone_candidates = [
            idx for idx in range(len(records))
            if idx != source_idx and records[idx]["ssn"] != source_record["ssn"]
        ]
        target_idx = rng.choice(clone_candidates)
        target_record = records[target_idx]

        target_record["ssn"] = source_record["ssn"]
        while target_record["full_name"] == source_record["full_name"]:
            target_record["full_name"] = fake.name()
        target_record["is_fraud"] = 1


def inject_credit_income_mismatch(record: dict[str, object]) -> None:
    # Fraud signal 5:
    # A very strong credit score paired with unusually low income, or a very
    # weak score paired with unusually high income, can signal manipulated
    # bureau data or inconsistent self-reported application details.
    if random.random() < 0.5:
        record["credit_score"] = random.randint(790, 850)
        record["annual_income"] = random.randint(18_000, 32_000)
    else:
        record["credit_score"] = random.randint(520, 565)
        record["annual_income"] = random.randint(150_000, 230_000)
    record["is_fraud"] = 1


def build_dataset(num_records: int = NUM_RECORDS) -> list[dict[str, object]]:
    records = [build_legit_record(applicant_index=i + 1) for i in range(num_records)]

    fraud_indices = list(range(num_records))
    random.shuffle(fraud_indices)
    cursor = 0

    def take_slice(count: int) -> list[int]:
        nonlocal cursor
        chosen = fraud_indices[cursor:cursor + count]
        cursor += count
        return chosen

    for idx in take_slice(650):
        inject_ssn_birth_year_mismatch(records[idx])
    for idx in take_slice(900):
        inject_unrealistic_income_to_loan_ratio(records[idx])
    for idx in take_slice(700):
        inject_short_stability(records[idx])
    for idx in take_slice(850):
        inject_credit_income_mismatch(records[idx])

    inject_duplicate_ssn(records, random.Random(SEED + 99))

    return records


def validate_records(records: list[dict[str, object]]) -> None:
    if len(records) != NUM_RECORDS:
        raise ValueError(f"Expected {NUM_RECORDS} records, found {len(records)}")

    required_fields = {
        "applicant_id",
        "full_name",
        "ssn",
        "date_of_birth",
        "address",
        "employer_name",
        "annual_income",
        "credit_score",
        "loan_amount",
        "loan_purpose",
        "email",
        "phone_number",
        "time_at_address_months",
        "time_at_employer_months",
        "num_recent_inquiries",
        "is_fraud",
    }

    for record in records:
        missing = required_fields.difference(record.keys())
        if missing:
            raise ValueError(f"Record missing required fields: {sorted(missing)}")

    if not any(not ssn_matches_birth_year(r["ssn"], int(str(r["date_of_birth"])[:4])) for r in records):
        raise ValueError("Expected at least one SSN/DOB mismatch record")


def save_dataset(records: list[dict[str, object]], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "applicant_id",
        "full_name",
        "ssn",
        "date_of_birth",
        "address",
        "employer_name",
        "annual_income",
        "credit_score",
        "loan_amount",
        "loan_purpose",
        "email",
        "phone_number",
        "time_at_address_months",
        "time_at_employer_months",
        "num_recent_inquiries",
        "is_fraud",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    records = build_dataset(NUM_RECORDS)
    validate_records(records)
    save_dataset(records, OUTPUT_PATH)
    fraud_count = sum(int(record["is_fraud"]) for record in records)
    print(f"Generated {len(records):,} records")
    print(f"Fraud-labelled records: {fraud_count:,}")
    print(f"Saved dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
