# F-ML-Data-Structuring
- [F-ML-Data-Structuring](#F-ML-Data-Structuring)
  - [Data](#Data)
  - [Bars](#Bars)


## Data
Sites that offer free tick data:  
- [Kibot](http://www.kibot.com/free_historical_data.aspx)
  - Offers tick data from 2009 to present for $IVE and $WDC
  - CSV file with fields: Date,Time,Price,Bid,Ask,Size
- TrueFX has currency data but couple of issues:
  - need a script to automate downloading all since it's separated by year and month
  - Also CSV looks to only have bid/ask, not actually tick price? Also no volume data either

## Bars
- TODO: make this all more clear
- Time bars: should avoid: pg 26
- Tick bars: exhibit desirable properties: may have Gaussian distrib (pg 27)
- Volume bars: ticks are arbitrary, vol even closer to IID Gaussian than tick. Lots of market microstructures revolve around price & volume
- Dollar bars: Variation of number of bars is much less when looking at market value exchanged. Number of shares also change multiple times of a security's life time.