# Credit application knowledge base

Use this document as an authoritative reference for collecting fields. These
rules describe the input schema; they are not a credit decision and must not
be presented as approval or rejection criteria.

## Income and debt

- `annual_inc`: gross annual income before taxes, in US dollars. If the
  applicant gives gross monthly income, multiply it by 12. Do not include
  guesses or net income unless the applicant explicitly says it is the only
  figure available.
- `monthly_debt_payments`: total monthly payments for existing credit
  obligations, in US dollars. Exclude rent, mortgage payments, and the new
  requested loan. This is an internal collection field and is not sent to the
  model as a final feature.
- `dti`: calculated locally as
  `100 * monthly_debt_payments / (annual_inc / 12)`. Never ask the applicant
  to provide DTI and never estimate it from incomplete values.

## Credit history

- `fico_range_low`: FICO score from 300 through 850. If the applicant gives
  one score, copy it to both `fico_range_low` and `fico_range_high`. If the
  applicant does not know or declines, leave both fields empty.
- `inq_last_6mths`: integer count of credit inquiries during the last six
  months, excluding automobile and mortgage inquiries. Use zero when the
  applicant explicitly says there were none.
- `mort_acc`: integer count of mortgage accounts. Use zero when the applicant
  explicitly says there are none.

## Housing and loan

- `home_ownership`: use exactly one of `RENT` (renter), `MORTGAGE` (homeowner
  with a mortgage), or `OWN` (homeowner without a mortgage).
- `loan_amnt`: requested loan amount in US dollars.
- `purpose`: use exactly one of `debt_consolidation`, `credit_card`,
  `home_improvement`, `major_purchase`, `car`, `medical`, `small_business`,
  `moving`, `vacation`, `house`, `wedding`, `renewable_energy`, or `other`.
  Ask a clarifying question if the purpose is ambiguous instead of guessing.

## Credit card usage

- `revol_util`: percentage of revolving credit limits currently used. For
  example, 2,000 dollars used out of a 10,000 dollar total limit is 20%.
  Accept values from 0 through 100 and ask for both used amount and total
  limit when the applicant does not know the percentage.

## Conversation safety

Ask exactly one question at a time. Extract only facts supported by the
applicant's latest message. If a value conflicts with an earlier answer, ask
which value is correct. Do not invent missing financial information, do not
give a final lending decision, and do not claim that this knowledge base is a
bank policy.

If the user's answer is only "yes", "yeah", "no", "none", or a number,
interpret it only in relation to the immediately preceding question.
If the answer cannot be mapped to a known field, ask a clarification question.

Only ask about these fields:
annual_inc
monthly_debt_payments
fico_range_low
home_ownership
inq_last_6mths
loan_amnt
mort_acc
purpose
revol_util

Never ask about:
employment
job
employer
age
marital status
nationality
education

## Monetary value normalization

The following inputs mean the same numeric value:

- `$20,000,000` -> `20000000`
- `20,000,000` -> `20000000`
- `20'000'000` -> `20000000`
- `20 000 000` -> `20000000`
- `20 million` -> `20000000`
- `20 millions` -> `20000000`
- `$20M` -> `20000000`
- `20k` -> `20000`
- `$3,500 per month` -> `monthly income = 3500`
- `3.5k monthly` -> `monthly income = 3500`

Remove currency symbols, commas, spaces, and apostrophes used as thousands
separators before converting a monetary value to a number.

Do not interpret a number as a loan amount unless the current target field is
`loan_amnt` or the user clearly states that the number is the requested loan
amount.

A negative answer to a count question is a valid answer, not an unknown
answer. Never treat "no" as missing when asking whether the applicant has any
accounts or inquiries.