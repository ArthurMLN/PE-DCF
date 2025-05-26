import seaborn as sns
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
import pytz

%run constant.ipynb

# 获取股票的历史数据
def get_historical_data(symbol, access_token, period_type='year', period=3, frequency_type='daily', frequency=1):
    base_url = 'https://api.schwabapi.com/marketdata/v1/pricehistory'
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    params = {
        'symbol': symbol,
        'periodType': period_type,
        'period': period,
        'frequencyType': frequency_type,
        'frequency': frequency,       
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Failed to fetch data for {symbol}: {response.status_code}")
        return None

    try:
        data = response.json()
        if 'candles' not in data or not data['candles']:
            print(f"No data found for {symbol}")
            return None
        df = pd.DataFrame(data['candles'])
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')  # 将datetime字段转换为日期时间格式
        return df
    except ValueError:
        print(f"Failed to parse JSON for {symbol}")
        return None

# 计算市场宽度
def calculate_market_breadth(sector_stocks, access_token):
    sector_breadth = pd.Series(dtype=float)
    for stock in sector_stocks:
        data = get_historical_data(stock, access_token)
        if data is not None and not data.empty:
            data['MA20'] = data['close'].rolling(window=20).mean()
            data = data.dropna()  # 去掉没有20日移动平均线的天数
            data['Above_MA20'] = data['close'] > data['MA20']
            sector_breadth = sector_breadth.add(data.set_index('datetime')['Above_MA20'], fill_value=0)
    
    if sector_breadth.empty:
        return None
    
    # 计算每个板块的平均分数
    sector_breadth = (sector_breadth / len(sector_stocks)) * 100
    return sector_breadth

# 假设我们有11个sector，每个sector包含若干股票
sectors = {    
    "XLV": ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "AMGN", "ISRG","PFE", "REGN", "ELV", "VRTX", "BSX", "SYK", "MDT", "BMY", "CI", "GILD","ZTS", "CVS", "HCA", "MCK", "BDX", "IQV", "HUM", "CNC", "A", "EW"],
    "XLI": ["GE", "CAT", "RTX", "UBER", "UNP", "HON", "LMT", "ETN", "ADP", "BA",
    "DE", "UPS", "TT", "WM", "PH", "TDG", "MMM", "NOC", "GD", "CTAS",
    "ITW", "CSX", "FDX", "EMR", "CARR", "NSC", "GEV", "PCAR", "URI", "JCI"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "LOW", "TJX", "BKNG", "SBUX", "NKE", "CMG",
    "ORLY", "AZO", "MAR", "DHI", "HLT", "ABNB", "GM", "ROST", "LEN", "F",
    "YUM", "RCL", "LULU", "EBAY", "TSCO", "NVR", "GRMN", "PHM", "DECK", "GPC"],
    "XLU": ["NEE", "SO", "DUK", "CEG", "AEP", "SRE", "D", "PEG", "PCG", "EXC",
    "ED", "XEL", "EIX", "WEC", "AWK", "VST", "DTE", "ETR", "ES", "PPL",
    "FE", "AEE", "CMS", "ATO", "NRG", "CNP", "LNT", "NI", "EVRG", "AES"],
    "XLB": ["LIN", "SHW", "FCX", "ECL", "APD", "NEM", "CTVA", "DOW", "NUE", "DD",
    "MLM", "VMC", "PPG", "IFF", "LYB", "SW", "BALL", "PKG", "AVY", "STLD",
    "IP", "AMCR", "CF", "CE", "EMN", "ALB", "MOS", "FMC"],
    "XLE": [  "XOM", "CVX", "EOG", "SLB", "MPC", "COP", "PSX", "WMB", "OKE", "VLO",
    "KMI", "OXY", "HES", "BKR", "FANG", "TRGP", "DVN", "HAL", "EQT", "CTRA",
    "MRO", "APA"],
    "XLF": ["JPM", "V", "MA", "BAC", "WFC", "GS", "SPGI", "AXP", "PGR",
    "MS", "BLK", "C", "CB", "MMC", "FI", "SCHW", "BX", "ICE", "KKR",
    "CME", "MCO", "AON", "PYPL", "PNC", "USB", "AJG", "TFC", "COF", "AFL"],
    "XLK": ["NVDA", "MSFT", "AAPL", "AVGO", "CRM", "ADBE", "AMD", "ORCL", "ACN", "CSCO",
    "QCOM", "INTU", "TXN", "IBM", "AMAT", "NOW", "MU", "LRCX", "ADI", "PANW",
    "KLAC", "INTC", "ANET", "SNPS", "APH", "CDNS", "MSI", "NXPI", "CRWD", "ROP"],
    "XLP": ["PG", "COST", "WMT", "KO", "PM", "PEP", "MDLZ", "MO", "CL", "TGT",
    "KMB", "KVUE", "GIS", "STZ", "SYY", "KDP", "KR", "MNST", "ADM", "HSY",
    "KHC", "DG", "CHD", "EL", "K", "DLTR", "MKC", "CLX", "TSN", "CAG"], 
    "XLRE": ["PLD", "AMT", "EQIX", "WELL", "O", "SPG", "PSA", "DLR", "CCI", "EXR",
    "CBRE", "VICI", "IRM", "AVB", "CSGP", "EQR", "VTR", "SBAC", "WY", "INVH",
    "ESS", "ARE", "MAA", "DOC", "KIM", "CPT", "UDR", "HST", "REG", "BXP"],
    "XLC": ["META", "GOOGL", "GOOG", "CHTR", "TMUS", "T", "EA", "CMCSA", "NFLX", "VZ",
    "DIS", "TTWO", "OMC", "WBD", "LYV", "IPG", "NWSA", "MTCH", "FOXA", "PARA", 
    "FOX", "NWS"]
}

access_token = get_new_tokens()

# 计算每个sector的市场宽度
sector_scores = {}
total_market_breadth = pd.Series(dtype=float)
for sector, stocks in sectors.items():
    sector_breadth = calculate_market_breadth(stocks, access_token)
    if sector_breadth is not None:
        sector_scores[sector] = sector_breadth
        total_market_breadth = total_market_breadth.add(sector_breadth, fill_value=0)

# 计算总市场宽度
if not total_market_breadth.empty:
    # 设置休斯顿时区
    houston_tz = pytz.timezone('America/Chicago')
    
    # 获取日期索引并转换为休斯顿时间
    total_market_breadth.index = total_market_breadth.index.tz_localize('UTC').tz_convert(houston_tz)
    # ✅ 保存为 Pickle 文件
    total_market_breadth.to_pickle("market_breadth_series.pkl")
    # 获取开始和结束日期
    start_date = total_market_breadth.index.min().strftime('%Y-%m-%d')
    end_date = total_market_breadth.index.max().strftime('%Y-%m-%d')

    # 绘制市场宽度图表
    plt.figure(figsize=(24, 6))
    plt.plot(total_market_breadth, label='Market Breadth')

    # 添加横线
    plt.axhline(900, color='red', linestyle='--', label='Extreme High (900)')
    plt.axhline(200, color='blue', linestyle='--', label='Extreme Low (200)')

    # 设置日期格式
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both', nbins=int(len(total_market_breadth) / 5)))
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

    plt.title(f'Market Breadth Over Time ({start_date} to {end_date})')
    plt.xlabel('Date (Houston Time)')
    plt.ylabel('Market Breadth Score')
    plt.xticks(rotation=45)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig('marketBreath-3Y.png')
    plt.show()
    
    # 转换sector_scores为DataFrame用于绘制热力图
    sector_scores_df = pd.DataFrame(sector_scores)
    
    # 将索引转换为日期格式
    sector_scores_df.index = sector_scores_df.index.tz_localize('UTC').tz_convert(houston_tz)
    sector_scores_df.index = pd.to_datetime(sector_scores_df.index).strftime('%Y-%m-%d')
    
    # 确保所有数据都是数字类型，并处理NaN值
    sector_scores_df = sector_scores_df.apply(pd.to_numeric, errors='coerce')
    sector_scores_df.fillna(0, inplace=True)  # 或者你可以使用 sector_scores_df.dropna()
    
    # 创建自定义的 colormap
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'white', 'red'], N=256)
    
    # 绘制热力图
    plt.figure(figsize=(24, 12))
    sns.heatmap(sector_scores_df.T, cmap=cmap, cbar=True, center=50)
    
    # 设置日期格式和间隔
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, prune='both', nbins=sector_scores_df.shape[1] // 5))
    plt.gca().set_xticklabels(sector_scores_df.columns, rotation=45)
    
    plt.title(f'Sector Market Breadth Heatmap ({start_date} to {end_date})')
    plt.xlabel('Date')
    plt.ylabel('Sector')
    plt.savefig('sectorHeatmap-3Y.png')
    plt.show()


else:
    print("No valid market breadth data to plot.")
