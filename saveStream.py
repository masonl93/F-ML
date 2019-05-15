import configparser
import datetime
import json
import queue
import time

from dateutil import tz
from dateutil.relativedelta import relativedelta

from InteractiveBrokers import InteractiveBrokers as ib
from InteractiveBrokers.Contracts import ContractSamples


def main(config):
    app = ib.App("127.0.0.1", int(config['DEFAULT']['port']), clientId=1)
    print("serverVersion:%s connectionTime:%s" % (app.client.serverVersion(),
                                                  app.client.twsConnectionTime()))
    contract = app.createContract(
        "SPY", "STK", "USD", "SMART", "ISLAND")
    app.IBreqTickByTickData(contract, "Last", 0, True)
    # create cron job?
    # how to run all day? Use cloud?
    try:
        while True:
            pass
    except KeyboardInterrupt:
        app.client.disconnect()
    return


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    main(config)
