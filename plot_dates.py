"""
This is an exploratory notebook showing some dates distributions. It is helpful
to better design our target, and is a good example of early data exploration.
"""
# %%
import pandas as pd
from skrub import TableReport
from credit_risk_models.risk_model_survival_analysis._loans import get_loans


loans = get_loans()

loans.loc[loans["is_ongoing"], "loan_duration"] = (
    pd.Timestamp.now() - loans.loc[loans["is_ongoing"]]["loan_created_date"] 
).dt.days

loans['reimbursment_duration'] = (
    loans['loan_reimbursed_date'] - loans['loan_created_date']
).dt.days

loans['termination_duration'] = (
    loans['terminated_at'] - loans['loan_created_date']
).dt.days

# %%

from matplotlib import pyplot as plt
import seaborn as sns

loans["raw_maturity_duration"] = (
    loans["raw_maturity_date"] - loans["loan_created_date"]
).dt.days
loans.sort_values("risks", inplace=True)

fig, ax = plt.subplots(dpi=300)
sns.scatterplot(
    loans,
    x="loan_created_date",
    y="raw_maturity_duration",
    hue="risks",
    ax=ax,
    size=3,
    alpha=.6,
)
labels = ax.get_xticklabels()
ticks = ax.get_xticks()
ax.axhline(y=180, color="k", linestyle="--", alpha=.8)
ax.axhline(y=150, color="k", linestyle="--", alpha=.4)
ax.set_xticks(ticks, labels=labels, rotation=50)
ax.set_ylabel("Maturity date (days)", size=12)
ax.set_xlabel("Loan created at", size=12)
plt.show()

# %%

fig, ax = plt.subplots(dpi=300)
sns.scatterplot(
    loans,
    x="reimbursment_duration",
    y="termination_duration",
    hue="risks",
    ax=ax,
    size=3,
    alpha=.6,
)
ax.plot([0, 290], [0, 290], color="k", linestyle="--", alpha=.4)
ax.axhline(y=150, color="k", linestyle="--", alpha=.4)
labels = ax.get_xticklabels()
ticks = ax.get_xticks()
ax.set_xticks(ticks, labels=labels, rotation=50)
ax.set_ylabel("Termination duration (days)", size=12)
ax.set_xlabel("Reimbursment duration (days)", size=12)
plt.show()

# %%
# DDs

from credit_risk_models.risk_model_survival_analysis._make_dataset import DatasetMaker

dd = DatasetMaker().due_diligences
dd["due_date_null"] = dd["dd_due_date"].isna()
print(dd.groupby("car_source")["due_date_null"].mean())

from skrub import TableReport
TableReport(dd.loc[dd["due_date_null"] & (dd["car_source"] == 0)])

# %%
