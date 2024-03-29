# Financial Machine Learning 
- [Financial Machine Learning](#financial-machine-learning)
  - [Data Analysis](#data-analysis)
    - [Data](#data)
    - [Structure](#structure)
    - [Bars (Ch 2)](#bars-ch-2)
    - [Labeling (Ch 3)](#labeling-ch-3)
    - [Sample Weights (Ch 4)](#sample-weights-ch-4)
  - [Notes](#notes)
  - [References](#references)

## Data Analysis

### Data
Sites that offer free tick data:  
- [Kibot](http://www.kibot.com/free_historical_data.aspx)
  - Offers tick data from 2009 to present for $IVE and $WDC
  - CSV file with fields: Date,Time,Price,Bid,Ask,Size  


### Structure
The file adv_fin_ml_func.py contains the functions from the book. Various modifications have been made such as Python 3 compatibility and better code formatting. Then there are folders for each "part" of the book. Under these folders are each chapter's exercises and various helper functions. For example, data_analysis/ contains chapters 2, 3, 4, and 5 end of chapter exercises.


### Bars (Ch 2)
- Time bars: should avoid by avoided (See pg 26)
  - Markets do not process information at a constant time interval
  - Exhibit poor statistical properties such as serial correlation and non-normality of returns
- Tick bars: (See pg 27)
  - Exhibit desirable properties: may have Gaussian distribution
- Volume bars: 
  - Ticks are arbitrary
  - Volume is even closer to IID Gaussian than tick.
  - Lots of market microstructures revolve around price & volume
- Dollar bars:
  - Variation of number of bars is much less when looking at market value exchanged.
  - Number of shares also change multiple times over a security's life time.


### Labeling (Ch 3)
- Triple Barrier Method
  - Label an observation according to the first barrier touched
  - Two horizontal barriers: profit-taking and stop-loss
    - Based of a dynamic function of estimated volatility
  - One vertical barrier: expiration limit 
    - Defined in terms of number of bars elapsed since position was taken
- Drop rare labels
  - Drop extremely rare labels since ML classifiers do not perform well when classes are too imbalanced
- Meta-labeling
  - Utilize a primary model (examples: trend following, ML algo, econometric equation, etc) to decide which side of a bet (long vs short) and use ML to decide the size of the bet, with zero size meaning no bet at all
  - Meta-labeling is helpful in achieving higher F1-scores
    - First, build a model that achieves high recall (true positive rate = TP/(TP+FN)), allowing for weak precision (TP/(TP+FP))
    - Second, correct the low precision by applying meta-labeling to the positives predicted by the first model
    - Filters out false positives, while primary model identifies the majority of positives
  - Main purpose is to determine if we should act or pass on the opportunity presented


### Sample Weights (Ch 4)
- Financial observations are not generated by independent and identically distributed (IID) processes
- Need sample weights to address this since most ML is based off of the IID assumption
- Begin by estimating the uniqueness of a label by determining if they are concurrent i.e. two labels are a function of at least one common return
- Markets are adaptive - as markets evolve, older examples are less relevant.
  - Samples weights should decay as new observations arise
- Sequential Bootstrap - draws are made according to a changing probability that controls for redundancy
- Weight observations by some function of uniqueness and absolute return since large absolute returns should be given more importance than labels with smaller absolute returns.
- Utilize class weights in addition to sample weights - weights that correct for underrepresented labels. 


## Notes
Please be aware that some mistakes may be present in the Jupyter notebooks. My Python knowledge is above average so the code should function properly with Python 3. My financial knowledge is at least average so there should not be many mistakes there. But my math skills are a work in progress so there might be some minor mistakes here. 


## References
Some other repos I found useful:
- https://github.com/BlackArbsCEO/Adv_Fin_ML_Exercises
- https://github.com/hudson-and-thames/mlfinlab
