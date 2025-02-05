import pandas as pd

# Load the CSV files
equity_df = pd.read_csv(r'd:\truthtell2\EQUITY_L.csv')
sme_equity_df = pd.read_csv(r'd:\truthtell2\SME_EQUITY_L.csv')
equity_bse_df = pd.read_csv(r'd:\truthtell2\Equity.csv')

# Rename the column 'Security Id' to 'SYMBOL'
equity_bse_df.rename(columns={'Security Id': 'SYMBOL'}, inplace=True)

# Define a function to append the suffix if not already appended
def append_suffix(symbol):
    while symbol.endswith(".NS"):
        symbol = symbol[:-3]
    return symbol + ".NS"

def append_bse_suffix(security_id):
    while str(security_id).endswith(".BO"):
        security_id = str(security_id)[:-3]
    return str(security_id) + ".BO"

# Apply the function to the Symbol column
equity_df['SYMBOL'] = equity_df['SYMBOL'].apply(append_suffix)
sme_equity_df['SYMBOL'] = sme_equity_df['SYMBOL'].apply(append_suffix)
equity_bse_df['SYMBOL'] = equity_bse_df['SYMBOL'].apply(append_bse_suffix)

# Save the modified dataframes back to CSV
equity_df.to_csv(r'd:\truthtell2\EQUITY_L.csv', index=False)
sme_equity_df.to_csv(r'd:\truthtell2\SME_EQUITY_L.csv', index=False)
equity_bse_df.to_csv(r'd:\truthtell2\Equity.csv', index=False)

# Display the first few rows of the modified dataframes
equity_df.head(), sme_equity_df.head(), equity_bse_df.head()
