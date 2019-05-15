import configparser
import datetime
import json
import queue
import time

from dateutil import tz
from dateutil.relativedelta import relativedelta

from InteractiveBrokers import InteractiveBrokers as ib
from InteractiveBrokers.Contracts import ContractSamples



to_zone = tz.tzlocal()
from_zone = tz.tzutc()
def localizeTime(ts):
    utc = datetime.datetime.utcfromtimestamp(ts)
    utc = utc.replace(tzinfo=from_zone)
    return utc.astimezone(to_zone).strftime("%Y%m%d %H:%M:%S")


def saveTickData(app, contract):
    start_time = "20190204 00:00:00"
    ticks = []
    # ohlc = []
    data = {'date': [], 'price': []}
    while True:
        print("requesting 1000 ticks")
        app.getHistTicks(contract, start_time)
        while True:
            if len(ticks) == 1000:
                break
            try:
                tick = app.tick_data_q.get(block=False)
                ticks.append(tick)
                print(localizeTime(tick.time))
            except queue.Empty:
                pass
        # with q.mutex:
        #     q.queue.clear()

        close_price = ticks[-1].price
        date = ticks[-1].time
        app.tick_data_q.queue.clear()
        ticks = []
        data['date'].append(date)
        data['price'].append(close_price)
        start_time = localizeTime(date)
        print(start_time)
        if start_time.startswith("20190304"):
            break
    print(data)
    with open('ticks.json', 'w') as f:
        json.dump(data, f)

def streamTicks(app, contract):
    app.IBreqTickByTickData(contract, "Last", 0, True)


def saveHistData(app, contract):
    app.getHistoricalData(contract, "1 Y", "ADJUSTED_LAST")
    import time
    time.sleep(2)
    data, reqId = app.hist_data_q.get(block=False)
    symbol = app.reqId_map[reqId]
    # for idx, row in data.iterrows():
    #     if row.date.startswith('201901'):
    #         print('jan')
    #         print(idx)
    #     elif row.date.startswith('201902'):
    #         print('feb')
    #         print(idx)
    data.to_csv('time_data.csv')


def main(config):
    # app = ib.App("127.0.0.1", args.port, clientId=1)
    app = ib.App("127.0.0.1", int(config['DEFAULT']['port']), clientId=1)
    print("serverVersion:%s connectionTime:%s" % (app.client.serverVersion(),
                                                  app.client.twsConnectionTime()))
    contract = app.createContract(
        "SPY", "STK", "USD", "SMART", "ISLAND")
    # saveTickData(app, contract)
    # saveHistData(app, contract)
    streamTicks(app, contract)
    while True:
        # create cron job?
        # or keep going until keyboard interrupt?
        # how to run all day? Use cloud?
        pass

    app.client.disconnect()
    return


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    main(config)


'''
TODO

Stream tick data live during market hours, start collection my own tick data for SPY/etc
Should be able to run on a cron similar code to that in jupyter ws with tick data each day
Save tick data to csv files for later processing
"No more than 1 tick-by-tick request can be made for the same instrument within 15 seconds."

'''