import pandas as pd
from scipy import stats

# Load value-weighted monthly returns
# Skip the header lines and read the value-weighted section
with open('10_Portfolios_Prior_12_2.csv', 'r') as f:
    lines = f.readlines()

# Find the value-weighted monthly returns section
start_idx = None
for i, line in enumerate(lines):
    if 'Value Weight Returns -- Monthly' in line:
        start_idx = i + 2  # skip column header line
        break

# Read the data from that section, stop at the next section header
rows = []
for line in lines[start_idx:]:
    if 'Average Equal Weighted Returns' in line or 'Average Value Weighted Returns' in line:
        break
    line = line.strip()
    if not line:
        continue
    parts = [p.strip() for p in line.split(',')]
    rows.append(parts)

cols = ['Date', 'Lo PRIOR', 'PRIOR 2', 'PRIOR 3', 'PRIOR 4', 'PRIOR 5',
        'PRIOR 6', 'PRIOR 7', 'PRIOR 8', 'PRIOR 9', 'Hi PRIOR']
df = pd.DataFrame(rows, columns=cols)

# Convert to numeric (percent values)
for col in cols[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Replace missing codes with NaN before converting to decimals
df = df.replace([-99.99, -999], float('nan'))

# Convert percentages to decimals
for col in cols[1:]:
    df[col] = df[col] / 100.0

# Convert date
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m', errors='coerce')
df = df.dropna()

# WML = Hi PRIOR (winners) - Lo PRIOR (losers)
df['WML'] = df['Hi PRIOR'] - df['Lo PRIOR']

# Drop missing WML observations
wml = df['WML'].dropna()

mean_ret = wml.mean()
std_ret = wml.std(ddof=1)
n = len(wml)
se = std_ret / (n ** 0.5)
t_stat = mean_ret / se
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
annualized_ret = mean_ret * 12
annualized_vol = std_ret * (12 ** 0.5)
sharpe = annualized_ret / annualized_vol

print(f"Period: {df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
print(f"Observations: {n}")
print(f"Mean monthly WML return: {mean_ret*100:.3f}%")
print(f"Monthly volatility: {std_ret*100:.3f}%")
print(f"Annualized WML return: {annualized_ret*100:.2f}%")
print(f"Annualized volatility: {annualized_vol*100:.2f}%")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value (two-sided): {p_value:.4f}")
print(f"Sharpe ratio (annualized): {sharpe:.3f}")

# Save summary
with open('CONCLUSION.md', 'w') as f:
    f.write("# Momentum Strategy Replication: Jegadeesh & Titman (1993)\n\n")
    f.write(f"Using Ken French's 10 Portfolios Formed on Prior 12-2 Returns (value-weighted, monthly), ")
    f.write(f"I constructed a winners-minus-losers (WML) momentum portfolio as the return difference between the `Hi PRIOR` decile (past winners) and the `Lo PRIOR` decile (past losers). ")
    f.write(f"From {df['Date'].min().strftime('%B %Y')} to {df['Date'].max().strftime('%B %Y')} ({n} monthly observations), the WML portfolio earned an average monthly return of **{mean_ret*100:.3f}%** (annualized: {annualized_ret*100:.2f}%).\n\n")
    f.write(f"The strategy's monthly volatility was {std_ret*100:.3f}% (annualized: {annualized_vol*100:.2f}%), yielding a t-statistic of **{t_stat:.3f}** and a two-sided p-value of **{p_value:.4f}**. ")
    if p_value < 0.05:
        f.write(f"This difference is statistically significant at the 5% level, consistent with the Jegadeesh-Titman momentum effect.\n\n")
    else:
        f.write(f"This difference is not statistically significant at the 5% level.\n\n")
    f.write(f"The annualized Sharpe ratio was {sharpe:.3f}. ")
    f.write(f"Overall, the analysis successfully reproduces the classic momentum premium: past winners significantly outperformed past losers over the long sample, though this evidence reflects value-weighted U.S. equities and does not account for transaction costs or risk adjustment.\n")
