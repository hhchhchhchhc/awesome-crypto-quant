# Awesome Crypto Quant [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of resources for cryptocurrency quantitative trading - libraries, frameworks, strategies, research papers, and more.

**Star this repo if you find it useful!** Contributions are welcome.

[English](#contents) | [中文](#中文版)

---

## Contents

- [Trading Frameworks](#trading-frameworks)
- [Data Sources](#data-sources)
- [Technical Analysis](#technical-analysis)
- [Machine Learning](#machine-learning)
- [Backtesting](#backtesting)
- [Execution & Order Management](#execution--order-management)
- [Risk Management](#risk-management)
- [Portfolio Optimization](#portfolio-optimization)
- [DeFi & On-Chain Analytics](#defi--on-chain-analytics)
- [High-Frequency Trading](#high-frequency-trading)
- [Market Making](#market-making)
- [Arbitrage](#arbitrage)
- [Sentiment Analysis](#sentiment-analysis)
- [Research & Papers](#research--papers)
- [Books](#books)
- [Courses & Tutorials](#courses--tutorials)
- [Communities](#communities)
- [Podcasts & YouTube](#podcasts--youtube)
- [Tools & Utilities](#tools--utilities)

---

## Trading Frameworks

### Python
| Project | Description | Stars |
|---------|-------------|-------|
| [CCXT](https://github.com/ccxt/ccxt) | Cryptocurrency exchange trading library with 100+ exchanges | ![GitHub stars](https://img.shields.io/github/stars/ccxt/ccxt) |
| [Freqtrade](https://github.com/freqtrade/freqtrade) | Free, open source crypto trading bot | ![GitHub stars](https://img.shields.io/github/stars/freqtrade/freqtrade) |
| [Hummingbot](https://github.com/hummingbot/hummingbot) | Open source market making & arbitrage bot | ![GitHub stars](https://img.shields.io/github/stars/hummingbot/hummingbot) |
| [Jesse](https://github.com/jesse-ai/jesse) | Advanced crypto trading framework for Python | ![GitHub stars](https://img.shields.io/github/stars/jesse-ai/jesse) |
| [OctoBot](https://github.com/Drakkar-Software/OctoBot) | Modular trading bot with AI capabilities | ![GitHub stars](https://img.shields.io/github/stars/Drakkar-Software/OctoBot) |
| [Catalyst](https://github.com/scrtlabs/catalyst) | Algorithmic trading library for crypto (Zipline-based) | ![GitHub stars](https://img.shields.io/github/stars/scrtlabs/catalyst) |
| [Cryptofeed](https://github.com/bmoscon/cryptofeed) | Cryptocurrency exchange feed handler | ![GitHub stars](https://img.shields.io/github/stars/bmoscon/cryptofeed) |
| [VNPY](https://github.com/vnpy/vnpy) | Python-based open source trading platform | ![GitHub stars](https://img.shields.io/github/stars/vnpy/vnpy) |
| [Backtrader](https://github.com/mementum/backtrader) | Python backtesting library for trading strategies | ![GitHub stars](https://img.shields.io/github/stars/mementum/backtrader) |

### Rust
| Project | Description | Stars |
|---------|-------------|-------|
| [Barter](https://github.com/barter-rs/barter-rs) | High-performance trading engine in Rust | ![GitHub stars](https://img.shields.io/github/stars/barter-rs/barter-rs) |
| [Galoy](https://github.com/GaloyMoney/galoy) | Bitcoin banking infrastructure | ![GitHub stars](https://img.shields.io/github/stars/GaloyMoney/galoy) |

### JavaScript/TypeScript
| Project | Description | Stars |
|---------|-------------|-------|
| [Superalgos](https://github.com/Superalgos/Superalgos) | Open-source crypto trading bot platform | ![GitHub stars](https://img.shields.io/github/stars/Superalgos/Superalgos) |
| [Gekko](https://github.com/askmike/gekko) | Bitcoin trading bot (archived but educational) | ![GitHub stars](https://img.shields.io/github/stars/askmike/gekko) |

---

## Data Sources

### Free APIs
| Source | Description | Type |
|--------|-------------|------|
| [Binance API](https://binance-docs.github.io/apidocs/) | Spot, Futures, Options data | REST/WebSocket |
| [CoinGecko API](https://www.coingecko.com/en/api) | Market data, 10k+ coins | REST |
| [CryptoCompare](https://min-api.cryptocompare.com/) | Historical & real-time data | REST/WebSocket |
| [Glassnode](https://glassnode.com/) | On-chain metrics | REST |
| [Messari](https://messari.io/api) | Crypto research & data | REST |
| [Alternative.me](https://alternative.me/crypto/fear-and-greed-index/) | Fear & Greed Index | REST |

### Data Libraries
| Project | Description | Stars |
|---------|-------------|-------|
| [Tardis.dev](https://github.com/tardis-dev/tardis-python) | Historical market data replay | ![GitHub stars](https://img.shields.io/github/stars/tardis-dev/tardis-python) |
| [CryptoDataDownload](https://www.cryptodatadownload.com/) | Free historical OHLCV data | CSV |
| [Kaiko](https://www.kaiko.com/) | Institutional-grade market data | API |

### On-Chain Data
| Source | Description |
|--------|-------------|
| [Dune Analytics](https://dune.com/) | SQL-based blockchain analytics |
| [Nansen](https://www.nansen.ai/) | Wallet labeling & smart money tracking |
| [Arkham](https://www.arkhamintelligence.com/) | Entity-based blockchain intelligence |
| [DefiLlama](https://defillama.com/) | DeFi TVL & protocol data |
| [Token Terminal](https://tokenterminal.com/) | Fundamental crypto metrics |

---

## Technical Analysis

| Project | Description | Stars |
|---------|-------------|-------|
| [TA-Lib](https://github.com/mrjbq7/ta-lib) | Technical analysis library wrapper | ![GitHub stars](https://img.shields.io/github/stars/mrjbq7/ta-lib) |
| [Pandas-TA](https://github.com/twopirllc/pandas-ta) | 130+ technical indicators for Pandas | ![GitHub stars](https://img.shields.io/github/stars/twopirllc/pandas-ta) |
| [FinTA](https://github.com/peerchemist/finta) | Financial technical analysis in Pandas | ![GitHub stars](https://img.shields.io/github/stars/peerchemist/finta) |
| [TradingView](https://www.tradingview.com/pine-script-docs/) | Pine Script for custom indicators | - |
| [Tulipy](https://github.com/cirla/tulipy) | Python bindings for Tulip Indicators | ![GitHub stars](https://img.shields.io/github/stars/cirla/tulipy) |

---

## Machine Learning

### Libraries & Frameworks
| Project | Description | Stars |
|---------|-------------|-------|
| [QLib](https://github.com/microsoft/qlib) | Microsoft's AI-oriented quantitative platform | ![GitHub stars](https://img.shields.io/github/stars/microsoft/qlib) |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | Deep reinforcement learning for trading | ![GitHub stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinRL) |
| [TensorTrade](https://github.com/tensortrade-org/tensortrade) | RL-based trading framework | ![GitHub stars](https://img.shields.io/github/stars/tensortrade-org/tensortrade) |
| [PFRL](https://github.com/pfnet/pfrl) | Deep RL library by Preferred Networks | ![GitHub stars](https://img.shields.io/github/stars/pfnet/pfrl) |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | Reliable RL implementations | ![GitHub stars](https://img.shields.io/github/stars/DLR-RM/stable-baselines3) |

### Feature Engineering
| Project | Description | Stars |
|---------|-------------|-------|
| [tsfresh](https://github.com/blue-yonder/tsfresh) | Automatic time series feature extraction | ![GitHub stars](https://img.shields.io/github/stars/blue-yonder/tsfresh) |
| [Featuretools](https://github.com/alteryx/featuretools) | Automated feature engineering | ![GitHub stars](https://img.shields.io/github/stars/alteryx/featuretools) |
| [TA-Lib](https://ta-lib.org/) | 200+ technical indicators | - |

### Models for Time Series
| Model | Use Case | Library |
|-------|----------|---------|
| LSTM/GRU | Sequential patterns | PyTorch/TensorFlow |
| Transformer | Long-range dependencies | Hugging Face |
| XGBoost/LightGBM | Tabular features | Native |
| Temporal Fusion Transformer | Multi-horizon forecasting | PyTorch Forecasting |
| N-BEATS | Time series forecasting | Darts |

---

## Backtesting

| Project | Description | Stars |
|---------|-------------|-------|
| [Vectorbt](https://github.com/polakowo/vectorbt) | Fast vectorized backtesting | ![GitHub stars](https://img.shields.io/github/stars/polakowo/vectorbt) |
| [Backtesting.py](https://github.com/kernc/backtesting.py) | Lightweight backtesting framework | ![GitHub stars](https://img.shields.io/github/stars/kernc/backtesting.py) |
| [Zipline-Reloaded](https://github.com/stefan-jansen/zipline-reloaded) | Pythonic algorithmic trading | ![GitHub stars](https://img.shields.io/github/stars/stefan-jansen/zipline-reloaded) |
| [Lean](https://github.com/QuantConnect/Lean) | QuantConnect's open-source engine | ![GitHub stars](https://img.shields.io/github/stars/QuantConnect/Lean) |
| [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader) | High-performance trading platform | ![GitHub stars](https://img.shields.io/github/stars/nautechsystems/nautilus_trader) |

### Backtesting Best Practices
- Always account for **slippage** (0.1-0.5% per trade)
- Include **trading fees** (maker: 0.02-0.1%, taker: 0.04-0.1%)
- Use **walk-forward optimization** to avoid overfitting
- Test on **out-of-sample** data
- Consider **funding rates** for perpetual futures

---

## Execution & Order Management

| Project | Description | Stars |
|---------|-------------|-------|
| [CCXT Pro](https://github.com/ccxt/ccxt) | Unified WebSocket API for exchanges | ![GitHub stars](https://img.shields.io/github/stars/ccxt/ccxt) |
| [Execution Algorithms](https://github.com/stefan-jansen/machine-learning-for-trading) | TWAP, VWAP implementations | - |

### Order Types
- **Limit Orders** - Reduce slippage, earn maker rebates
- **TWAP** - Time-weighted average price
- **VWAP** - Volume-weighted average price
- **Iceberg** - Hidden large orders
- **Post-Only** - Ensure maker fee

---

## Risk Management

### Libraries
| Project | Description | Stars |
|---------|-------------|-------|
| [Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | Portfolio optimization & risk | ![GitHub stars](https://img.shields.io/github/stars/dcajasn/Riskfolio-Lib) |
| [PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) | Mean-variance optimization | ![GitHub stars](https://img.shields.io/github/stars/robertmartin8/PyPortfolioOpt) |
| [empyrical](https://github.com/quantopian/empyrical) | Financial risk metrics | ![GitHub stars](https://img.shields.io/github/stars/quantopian/empyrical) |

### Key Metrics
```python
# Common Risk Metrics
- Sharpe Ratio = (Return - Risk_Free) / Volatility
- Sortino Ratio = (Return - Risk_Free) / Downside_Volatility
- Max Drawdown = (Peak - Trough) / Peak
- Value at Risk (VaR) = Quantile of loss distribution
- Calmar Ratio = Annual Return / Max Drawdown
```

### Position Sizing
- **Kelly Criterion**: `f* = (bp - q) / b`
- **Fixed Fractional**: Risk X% of capital per trade
- **Volatility Targeting**: Scale position by inverse volatility

---

## Portfolio Optimization

| Method | Description | Library |
|--------|-------------|---------|
| Mean-Variance (Markowitz) | Classic optimization | PyPortfolioOpt |
| Risk Parity | Equal risk contribution | Riskfolio-Lib |
| Black-Litterman | Incorporate views | PyPortfolioOpt |
| Hierarchical Risk Parity | ML-based clustering | Riskfolio-Lib |
| CVaR Optimization | Tail risk focus | cvxpy |

---

## DeFi & On-Chain Analytics

### MEV & Arbitrage
| Project | Description | Stars |
|---------|-------------|-------|
| [Flashbots](https://github.com/flashbots/mev-boost) | MEV extraction infrastructure | ![GitHub stars](https://img.shields.io/github/stars/flashbots/mev-boost) |
| [Artemis](https://github.com/paradigmxyz/artemis) | MEV bot framework by Paradigm | ![GitHub stars](https://img.shields.io/github/stars/paradigmxyz/artemis) |
| [MEV-Share](https://github.com/flashbots/mev-share) | MEV redistribution | ![GitHub stars](https://img.shields.io/github/stars/flashbots/mev-share) |

### DeFi Protocols
| Project | Description | Stars |
|---------|-------------|-------|
| [Uniswap V3](https://github.com/Uniswap/v3-core) | Concentrated liquidity AMM | ![GitHub stars](https://img.shields.io/github/stars/Uniswap/v3-core) |
| [Aave](https://github.com/aave/aave-v3-core) | Lending protocol | ![GitHub stars](https://img.shields.io/github/stars/aave/aave-v3-core) |
| [Compound](https://github.com/compound-finance/compound-protocol) | Money markets | ![GitHub stars](https://img.shields.io/github/stars/compound-finance/compound-protocol) |

### Analytics Tools
| Tool | Description |
|------|-------------|
| [EigenPhi](https://eigenphi.io/) | MEV & arbitrage tracker |
| [Cielo Finance](https://cielo.finance/) | Wallet tracker |
| [Zerion](https://zerion.io/) | DeFi portfolio tracker |

---

## High-Frequency Trading

### Infrastructure
| Component | Options |
|-----------|---------|
| Language | C++, Rust, Python (numpy/numba) |
| Networking | Kernel bypass (DPDK), FPGA |
| Co-location | Exchange proximity hosting |
| Market Data | Direct feed, normalized |

### Latency Optimization
```
Exchange API → < 1ms
Order placement → < 100μs (co-located)
Market data processing → < 10μs
Strategy decision → < 1μs
```

### Libraries
| Project | Description | Stars |
|---------|-------------|-------|
| [Arctic](https://github.com/man-group/arctic) | High-performance time series DB | ![GitHub stars](https://img.shields.io/github/stars/man-group/arctic) |
| [Polars](https://github.com/pola-rs/polars) | Fast DataFrame library | ![GitHub stars](https://img.shields.io/github/stars/pola-rs/polars) |
| [Numba](https://github.com/numba/numba) | JIT compiler for Python | ![GitHub stars](https://img.shields.io/github/stars/numba/numba) |

---

## Market Making

### Concepts
- **Bid-Ask Spread**: Your profit margin
- **Inventory Risk**: Holding unwanted positions
- **Adverse Selection**: Trading against informed traders

### Strategies
| Strategy | Description |
|----------|-------------|
| Avellaneda-Stoikov | Optimal quoting with inventory |
| Grid Trading | Orders at fixed intervals |
| Ping Pong | Tight spread, high frequency |

### Resources
- [Hummingbot Academy](https://hummingbot.org/academy/) - Market making tutorials
- [Avellaneda-Stoikov Paper](https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf) - Original paper

---

## Arbitrage

### Types
| Type | Description | Difficulty |
|------|-------------|------------|
| Spot-Spot | Price difference across exchanges | Medium |
| Spot-Futures | Basis trading | Medium |
| Triangular | A→B→C→A within exchange | High |
| Cross-DEX | AMM price differences | High (MEV) |
| Funding Rate | Long spot, short perp | Low |

### Considerations
- Transaction fees eat profits
- Withdrawal times & limits
- Capital lockup on multiple exchanges
- Counterparty risk

---

## Sentiment Analysis

### Data Sources
| Source | Type |
|--------|------|
| [LunarCrush](https://lunarcrush.com/) | Social metrics |
| [Santiment](https://santiment.net/) | On-chain + social |
| [The TIE](https://www.thetie.io/) | Institutional sentiment |
| Twitter/X API | Real-time tweets |
| Reddit API | r/cryptocurrency, r/bitcoin |

### NLP Libraries
| Project | Description | Stars |
|---------|-------------|-------|
| [FinBERT](https://github.com/ProsusAI/finBERT) | Financial sentiment BERT | ![GitHub stars](https://img.shields.io/github/stars/ProsusAI/finBERT) |
| [Transformers](https://github.com/huggingface/transformers) | State-of-the-art NLP | ![GitHub stars](https://img.shields.io/github/stars/huggingface/transformers) |
| [VADER](https://github.com/cjhutto/vaderSentiment) | Rule-based sentiment | ![GitHub stars](https://img.shields.io/github/stars/cjhutto/vaderSentiment) |

---

## Research & Papers

### Must-Read Papers
| Paper | Topic | Year |
|-------|-------|------|
| [Deep Learning for Limit Order Books](https://arxiv.org/abs/1601.01987) | LOB prediction | 2016 |
| [Bitcoin Price Prediction with RL](https://arxiv.org/abs/1810.01239) | Reinforcement learning | 2018 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformer architecture | 2017 |
| [Flash Boys 2.0](https://arxiv.org/abs/1904.05234) | DEX frontrunning | 2019 |
| [Avellaneda-Stoikov](https://www.math.nyu.edu/~avellane/HighFrequencyTrading.pdf) | Market making | 2008 |

### Research Repositories
| Repository | Description | Stars |
|------------|-------------|-------|
| [Papers With Code - Trading](https://paperswithcode.com/task/algorithmic-trading) | ML trading papers | - |
| [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta) | Financial RL research | ![GitHub stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinRL-Meta) |

---

## Books

### Quantitative Trading
| Book | Author | Level |
|------|--------|-------|
| Advances in Financial ML | Marcos López de Prado | Advanced |
| Machine Learning for Asset Managers | Marcos López de Prado | Advanced |
| Algorithmic Trading | Ernest Chan | Intermediate |
| Quantitative Trading | Ernest Chan | Beginner |
| Python for Finance | Yves Hilpisch | Intermediate |
| Trading and Exchanges | Larry Harris | Intermediate |

### Crypto-Specific
| Book | Author | Level |
|------|--------|-------|
| Mastering Bitcoin | Andreas Antonopoulos | Intermediate |
| Mastering Ethereum | Andreas Antonopoulos | Intermediate |
| The Bitcoin Standard | Saifedean Ammous | Beginner |

---

## Courses & Tutorials

### Free Courses
| Course | Platform | Topic |
|--------|----------|-------|
| [QuantConnect Bootcamp](https://www.quantconnect.com/learning) | QuantConnect | Algo trading |
| [Freqtrade Docs](https://www.freqtrade.io/) | Freqtrade | Bot setup |
| [Hummingbot Academy](https://hummingbot.org/academy/) | Hummingbot | Market making |
| [CryptoZombies](https://cryptozombies.io/) | - | Solidity basics |

### Paid Courses
| Course | Platform | Topic |
|--------|----------|-------|
| ML for Trading | Coursera | Machine learning |
| Algorithmic Trading | Udacity | Python trading |
| Blockchain Specialization | Coursera | Crypto fundamentals |

### YouTube Channels
| Channel | Focus |
|---------|-------|
| [Part Time Larry](https://www.youtube.com/@parttimelarry) | Python trading bots |
| [Algo Trading 101](https://www.youtube.com/@AlgoTrading101) | Strategy development |
| [Benjamin Cowen](https://www.youtube.com/@intothecryptoverse) | Crypto analysis |

---

## Communities

### Discord & Telegram
| Community | Platform | Focus |
|-----------|----------|-------|
| Freqtrade | Discord | Bot development |
| Hummingbot | Discord | Market making |
| QuantConnect | Slack | Algo trading |
| Crypto Quant Traders | Telegram | General quant |

### Forums
| Forum | Focus |
|-------|-------|
| [r/algotrading](https://reddit.com/r/algotrading) | Algo trading discussion |
| [r/CryptoCurrency](https://reddit.com/r/CryptoCurrency) | Crypto general |
| [QuantConnect Forum](https://www.quantconnect.com/forum) | Strategy discussion |
| [Elite Trader](https://www.elitetrader.com/) | Professional trading |

---

## Podcasts & YouTube

### Podcasts
| Podcast | Focus |
|---------|-------|
| Chat With Traders | Professional traders |
| Flirting with Models | Quant strategies |
| Bankless | DeFi & crypto |
| The Pomp Podcast | Crypto investing |

---

## Tools & Utilities

### Development
| Tool | Description |
|------|-------------|
| [Jupyter](https://jupyter.org/) | Interactive notebooks |
| [VS Code](https://code.visualstudio.com/) | IDE |
| [Docker](https://www.docker.com/) | Containerization |
| [Git](https://git-scm.com/) | Version control |

### Monitoring
| Tool | Description |
|------|-------------|
| [Grafana](https://grafana.com/) | Metrics visualization |
| [Prometheus](https://prometheus.io/) | Time series database |
| [PagerDuty](https://www.pagerduty.com/) | Alerting |

### Cloud Providers
| Provider | Best For |
|----------|----------|
| AWS | Full infrastructure |
| DigitalOcean | Simple VPS |
| Hetzner | Cost-effective EU |
| Vultr | Low latency |

---

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

1. Fork this repository
2. Create your feature branch (`git checkout -b add-resource`)
3. Commit your changes (`git commit -am 'Add new resource'`)
4. Push to the branch (`git push origin add-resource`)
5. Create a Pull Request

---

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

---

<div align="center">

**If you found this helpful, please star the repo!**

Made with love for the crypto quant community

</div>

---

# 中文版

> 加密货币量化交易资源精选列表 - 框架、数据源、策略、论文等

## 目录

- [交易框架](#交易框架)
- [数据源](#数据源-1)
- [技术分析](#技术分析-1)
- [机器学习](#机器学习-1)
- [回测框架](#回测框架)
- [风险管理](#风险管理-1)
- [DeFi与链上分析](#defi与链上分析)
- [高频交易](#高频交易-1)
- [做市策略](#做市策略)
- [套利策略](#套利策略)
- [情绪分析](#情绪分析-1)
- [研究论文](#研究论文)
- [书籍推荐](#书籍推荐)
- [课程教程](#课程教程)
- [社区资源](#社区资源)

---

## 交易框架

### Python 框架
| 项目 | 描述 | 推荐度 |
|------|------|--------|
| [CCXT](https://github.com/ccxt/ccxt) | 统一交易所API，支持100+交易所 | ⭐⭐⭐⭐⭐ |
| [Freqtrade](https://github.com/freqtrade/freqtrade) | 开源加密货币交易机器人 | ⭐⭐⭐⭐⭐ |
| [VNPY](https://github.com/vnpy/vnpy) | 国产开源量化交易平台 | ⭐⭐⭐⭐⭐ |
| [Hummingbot](https://github.com/hummingbot/hummingbot) | 做市与套利机器人 | ⭐⭐⭐⭐ |
| [Jesse](https://github.com/jesse-ai/jesse) | 专业级加密货币交易框架 | ⭐⭐⭐⭐ |

---

## 数据源

### 免费数据
| 来源 | 描述 | 数据类型 |
|------|------|----------|
| [币安API](https://binance-docs.github.io/apidocs/) | 现货、合约、期权数据 | K线/深度/成交 |
| [CoinGecko](https://www.coingecko.com/en/api) | 市场数据，10000+币种 | 价格/市值/交易量 |
| [Glassnode](https://glassnode.com/) | 链上指标 | 链上数据 |

### 链上数据
| 来源 | 描述 |
|------|------|
| [Dune Analytics](https://dune.com/) | SQL查询区块链数据 |
| [Nansen](https://www.nansen.ai/) | 聪明钱追踪 |
| [DefiLlama](https://defillama.com/) | DeFi TVL数据 |

---

## 机器学习

### 框架推荐
| 项目 | 描述 | 适用场景 |
|------|------|----------|
| [QLib](https://github.com/microsoft/qlib) | 微软AI量化平台 | 因子挖掘/模型训练 |
| [FinRL](https://github.com/AI4Finance-Foundation/FinRL) | 深度强化学习交易 | 策略优化 |
| [LightGBM](https://github.com/microsoft/LightGBM) | 梯度提升框架 | 表格数据预测 |

### 特征工程
| 库 | 描述 |
|---|------|
| [tsfresh](https://github.com/blue-yonder/tsfresh) | 时序特征自动提取 |
| [Pandas-TA](https://github.com/twopirllc/pandas-ta) | 130+技术指标 |

---

## 回测框架

| 项目 | 描述 | 特点 |
|------|------|------|
| [Vectorbt](https://github.com/polakowo/vectorbt) | 向量化回测 | 速度快 |
| [Backtrader](https://github.com/mementum/backtrader) | 事件驱动回测 | 功能全 |
| [Nautilus Trader](https://github.com/nautechsystems/nautilus_trader) | 高性能交易平台 | 专业级 |

### 回测注意事项
```python
# 必须考虑的因素
1. 滑点 (Slippage): 0.1-0.5%
2. 手续费 (Fees): Maker 0.02%, Taker 0.05%
3. 资金费率 (Funding Rate): 每8小时
4. 样本外测试 (Out-of-Sample)
5. 过拟合检验 (Overfitting Check)
```

---

## 风险管理

### 常用指标
```python
# 风险指标计算
夏普比率 = (收益率 - 无风险利率) / 波动率
最大回撤 = (峰值 - 谷值) / 峰值
卡玛比率 = 年化收益 / 最大回撤
索提诺比率 = 收益率 / 下行波动率
```

### 仓位管理
| 方法 | 公式 | 适用场景 |
|------|------|----------|
| 凯利公式 | f = (bp-q)/b | 理论最优 |
| 固定比例 | 每次风险X%本金 | 保守型 |
| 波动率目标 | 仓位 ∝ 1/波动率 | 风险平价 |

---

## 做市策略

### 核心概念
- **买卖价差**: 利润来源
- **库存风险**: 持仓偏离
- **逆向选择**: 与知情交易者对手交易

### 经典模型
| 模型 | 描述 |
|------|------|
| Avellaneda-Stoikov | 最优报价模型 |
| 网格交易 | 固定间隔挂单 |
| 高频做市 | 极窄价差，高频交易 |

---

## 套利策略

| 类型 | 描述 | 难度 |
|------|------|------|
| 现货套利 | 交易所间价差 | ⭐⭐ |
| 期现套利 | 现货-合约基差 | ⭐⭐ |
| 三角套利 | A→B→C→A | ⭐⭐⭐ |
| 跨DEX套利 | AMM价差 | ⭐⭐⭐⭐ |
| 资金费率套利 | 多现货空永续 | ⭐ |

![32ed3917f1f7f20f3e561e3b386da68e](https://github.com/user-attachments/assets/e6ad42c2-3bd1-4426-a4ac-53458cf2ce84)

---

## 书籍推荐

### 量化交易
| 书名 | 作者 | 难度 |
|------|------|------|
| 《金融机器学习进阶》 | Marcos López de Prado | 高级 |
| 《算法交易》 | Ernest Chan | 中级 |
| 《Python金融实战》 | Yves Hilpisch | 中级 |

### 加密货币
| 书名 | 作者 | 难度 |
|------|------|------|
| 《精通比特币》 | Andreas Antonopoulos | 中级 |
| 《精通以太坊》 | Andreas Antonopoulos | 中级 |

---

## 社区资源

### 国内社区
| 社区 | 平台 | 主题 |
|------|------|------|
| 量化交易研究 | 微信群 | 策略讨论 |
| 币Quant | 知识星球 | 加密量化 |
| VNPY社区 | 论坛 | 框架使用 |

### 国际社区
| 社区 | 平台 | 主题 |
|------|------|------|
| r/algotrading | Reddit | 算法交易 |
| Freqtrade | Discord | 机器人开发 |
| Hummingbot | Discord | 做市策略 |

---

<div align="center">

**觉得有用请点Star支持！欢迎提交PR贡献资源！**

朋友们现在的Tardis API是solo 1200每月的档位，年付年付！（14400美元折合人民币10万多人民币）
现货、期货、期权的数据都有啦！需要数据的联系微信a13479004101，代下载每一T收费249，单次最低99。

致富证券可能是一张内地身份证5分钟线上开户美港股的最后一台车了。
开户奖励很丰厚，总价值高达HK$8,800，包括现金券、返现、股票和转仓回赠。推荐好友还能额外拿钱，上不封顶。具体奖励有：1. 开户直接拿HK$200现金券+交易券；2. 入金最高返HK$1,300，还送NVIDIA股票；3. 转仓最高回赠HK$7,000；4. 推荐好友每成功一个得HK$300。

注意需要通过我的邀请码XC6774（二维码：https://image2url.com/r2/default/images/1770021823559-f8b6c7cb-d63d-4b01-9ead-48455da15c08.png或者https://mp.weixin.qq.com/s/NnIA1hszlLCyQ6BVxTmMAw）注册才有数千额外奖励QAQ。

致富支持大陆用户无港卡开户 ，可尝试用工商银行等快速入金，需开户7天内入金1万港币等值的前才有上述奖励。

致富chief.apk
链接: https://pan.baidu.com/s/1gpI4JSYY6MwZqP8Ywor9Tw?pwd=hmec 提取码: hmec

![32ed3917f1f7f20f3e561e3b386da68e](https://github.com/user-attachments/assets/e6ad42c2-3bd1-4426-a4ac-53458cf2ce84)
![f2ded2c901a4e731b32d4addcce25f3c](https://github.com/user-attachments/assets/10d5fefd-a939-435a-b252-ec29f2bbbbea)
![微信图片_20260203141110_215_152](https://github.com/user-attachments/assets/bced6473-80cd-4569-9d20-44c46ffde382)
![微信图片_20260203141111_216_152](https://github.com/user-attachments/assets/320f2928-8201-4977-b81d-9dd47f94c48f)



</div>


