import sys

sys.path.append("../../")
import talib


class agg_feature:

    # ===================
    # Overlap Studies
    # ===================
    def BBANDS(df):
        # DIV close for feature_columns to use close in backtest
        feature_columns = []
        close = df["close"].values
        # BBANDS - Bollinger Bands
        df["upperband"], df["middleband"], df["lowerband"] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        df["BBANDS_ratio_1"] = df["upperband"] / df["middleband"]
        df["BBANDS_ratio_2"] = df["upperband"] / df["lowerband"]
        feature_columns += ["upperband", "middleband", "lowerband", "BBANDS_ratio_1", "BBANDS_ratio_2"]

        return df, feature_columns

    def DEMA(df):
        # DEMA - Double Exponential Moving Average
        for i in [5, 30, 60, 120]:
            feature_columns = []
            close = df["close"].values
            df[f"DEMA{i}"] = talib.DEMA(close, timeperiod=i)
            feature_columns += [f"DEMA{i}"]
            return df, feature_columns

    def HT_TRENDLINE(df):
        feature_columns = []
        close = df["close"].values
        #  - Hilbert Transform - Instantaneous Trendline
        df["HT_TRENDLINE"] = talib.HT_TRENDLINE(close)
        feature_columns += ["HT_TRENDLINE"]
        return df, feature_columns

    def KAMA(df):
        for i in [5, 30, 60, 120]:
            feature_columns = []
            close = df["close"].values
            df[f"KAMA{i}"] = talib.KAMA(close, timeperiod=i)
            feature_columns += [f"KAMA{i}"]
            return df, feature_columns

    def MAMA(df):
        feature_columns = []
        close = df["close"].values
        # MAMA - MESA Adaptive Moving Average
        mama, fama = talib.MAMA(close, fastlimit=0.1, slowlimit=0.01)
        df["MMA"] = mama / fama
        feature_columns += ["MMA"]
        return df, feature_columns

    def MIDPRICE(df):
        for i in [5, 30, 60, 120]:
            feature_columns = []
            high = df["high"].values
            low = df["low"].values
            df[f"MIDPRICE{i}"] = talib.MIDPRICE(high, low, timeperiod=14)
            feature_columns += [f"MIDPRICE{i}"]
            return df, feature_columns

    def SAR(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        # SAR - Parabolic SAR
        df["SAR"] = talib.SAR(high, low, acceleration=0, MINMAXINDEXimum=0)
        feature_columns += ["SAR"]
        return df, feature_columns

    def SAREXT(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        # SAREXT - Parabolic SAR - Extended
        df["SAREXT"] = talib.SAREXT(
            high,
            low,
            startvalue=0,
            offsetonreverse=0,
            accelerationinitlong=0,
            accelerationlong=0,
            accelerationMINMAXINDEXlong=0,
            accelerationinitshort=0,
            accelerationshort=0,
            accelerationMINMAXINDEXshort=0,
        )
        feature_columns += ["SAREXT"]
        return df, feature_columns

    def T3(df):
        for i in [5, 30, 60, 120]:
            feature_columns = []
            close = df["close"].values
            df[f"T3_{i}"] = talib.T3(close, timeperiod=i, vfactor=0)
            feature_columns += [f"T3_{i}"]
            return df, feature_columns

    def TEMA(df):
        for i in [5, 30, 60, 120]:
            feature_columns = []
            close = df["close"].values
            df[f"TEMA{i}"] = talib.TEMA(close, timeperiod=i)
            feature_columns += [f"TEMA{i}"]
            return df, feature_columns

    def TRIMA(df):
        for i in [5, 30, 60, 120]:
            feature_columns = []
            close = df["close"].values
            df[f"TRIMA{i}"] = talib.TRIMA(close, timeperiod=i)
            feature_columns += [f"TRIMA{i}"]
            return df, feature_columns

    def WMA(df):
        for i in [5, 30, 60, 120]:
            feature_columns = []
            close = df["close"].values
            df[f"WMA{i}"] = talib.WMA(close, timeperiod=i)
            feature_columns += [f"WMA{i}"]
            return df, feature_columns

    # ===================
    #  Momentum Indicator Functions
    # ===================

    def ADX(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["ADX"] = talib.ADX(high, low, close, timeperiod=14)
        feature_columns += ["ADX"]
        return df, feature_columns

    def ADXR(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["ADXR"] = talib.ADXR(high, low, close, timeperiod=14)
        feature_columns += ["ADXR"]
        return df, feature_columns

    def APO(df):
        feature_columns = []
        close = df["close"].values
        df["APO"] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
        feature_columns += ["APO"]
        return df, feature_columns

    def AROON(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        df["aroondown"], df["aroonup"] = talib.AROON(high, low, timeperiod=14)
        df["aroondown_aroonup_ratio"] = df["aroondown"] / df["aroonup"]
        feature_columns += ["aroondown", "aroonup", "aroondown_aroonup_ratio"]
        return df, feature_columns

    def AROONOSC(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        df["AROONOSC"] = talib.AROONOSC(high, low, timeperiod=14)
        feature_columns += ["AROONOSC"]
        return df, feature_columns

    def BOP(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["BOP"] = talib.BOP(open, high, low, close)
        feature_columns += ["BOP"]
        return df, feature_columns

    def CCI(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CCI"] = talib.CCI(high, low, close, timeperiod=14)
        feature_columns += ["CCI"]
        return df, feature_columns

    def CMO(df):
        close = df["close"].values
        feature_columns = []
        df["CMD"] = talib.CMO(close, timeperiod=14)
        feature_columns += ["CMD"]
        return df, feature_columns

    def DX(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["DX"] = talib.DX(high, low, close, timeperiod=14)
        feature_columns += ["DX"]
        return df, feature_columns

    def MAC(df):
        feature_columns = []
        close = df["close"].values
        # MACD - Moving Average Convergence/Divergence
        df["macd"], df["macdsignal"], df["macdhist"] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df["macd_macdsignal_ratio"] = df["macd"] / df["macdsignal"]
        df["macdsignal_macdhist_ratio"] = df["macdsignal"] / df["macdhist"]
        feature_columns += ["macd", "macdsignal", "macdhist", "macd_macdsignal_ratio", "macdsignal_macdhist_ratio"]
        return df, feature_columns

    def MACDEXT(df):
        feature_columns = []
        close = df["close"].values
        # MACDEXT - MACD with controllable MA type
        df["macd_dext"], df["macdsignal_dext"], df["macdhist_dext"] = talib.MACDEXT(
            close, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0
        )
        df["macd_macdsignal_ratio_dext"] = df["macd_dext"] / df["macdsignal_dext"]
        df["macdsignal_macdhist_ratio_dext"] = df["macdsignal_dext"] / df["macdhist_dext"]
        feature_columns += ["macd_dext", "macdsignal_dext", "macdhist_dext", "macd_macdsignal_ratio_dext", "macdsignal_macdhist_ratio_dext"]
        return df, feature_columns

    def MACDFIX(df):
        feature_columns = []
        close = df["close"].values
        # MACDFIX - Moving Average Convergence/Divergence Fix 12/26
        df["macd_fix"], df["macdsignal_fix"], df["macdhist_fix"] = talib.MACDFIX(close, signalperiod=9)
        df["macd_macdsignal_ratio_fix"] = df["macd_fix"] / df["macdsignal_fix"]
        df["macdsignal_macdhist_ratio_fix"] = df["macdsignal_fix"] / df["macdhist_fix"]
        feature_columns += ["macd_fix", "macdsignal_fix", "macd_macdsignal_ratio_fix", "macdsignal_macdhist_ratio_fix"]

        return df, feature_columns

    def MI(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        volume = df["volume"].values

        # MFI - Money Flow Index
        df["MFI"] = talib.MFI(high, low, close, volume, timeperiod=14)
        feature_columns += ["MFI"]

        # MINMAXUS_DI - MinMAXus Directional Indicator
        # df["MINMAXUS_DI"] = talib.MINMAXUS_DI(high, low, close, timeperiod=14)
        # feature_columns += ["MINMAXUS_DI"]

        # MINMAXUS_DM - MinMAXus Directional Movement
        # df["MINMAXUS_DM"] = talib.MINMAXUS_DM(high, low, timeperiod=14)
        # feature_columns += ["MINMAXUS_DM"]

        # MOM - Momentum
        df["MOM"] = talib.MOM(close, timeperiod=10)
        feature_columns += ["MOM"]

        # PLUS_DI - Plus Directional Indicator
        df["PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
        feature_columns += ["PLUS_DI"]

        # PLUS_DM - Plus Directional Movement
        df["PLUS_DM"] = talib.PLUS_DM(high, low, timeperiod=14)
        feature_columns += ["PLUS_DM"]

        return df, feature_columns

    def RO(df):
        feature_columns = []
        close = df["close"].values
        # PPO - Percentage Price Oscillator
        df["PPO"] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        feature_columns += ["PPO"]

        # ROC - Rate of change : ((price/prevPrice)-1)*100
        for i in [5, 30, 60, 120]:
            df[f"ROC_{i}"] = talib.ROC(close, timeperiod=i)
            feature_columns += [f"ROC_{i}"]

        # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
        for i in [5, 30, 60, 120]:
            df[f"ROCP_{i}"] = talib.ROCP(close, timeperiod=10)
            feature_columns += [f"ROCP_{i}"]
        return df, feature_columns

    def ROCR(df):
        feature_columns = []
        close = df["close"].values
        # ROCR - Rate of change ratio: (price/prevPrice)
        for i in [5, 30, 60, 120]:
            df[f"ROCR_{i}"] = talib.ROCR(close, timeperiod=i)
            feature_columns += [f"ROCR_{i}"]

        # ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
        for i in [5, 30, 60, 120]:
            df[f"ROCR100_{i}"] = talib.ROCR100(close, timeperiod=i)
            feature_columns += [f"ROCR100_{i}"]

        # RSI - Relative Strength Index
        df["RSI"] = talib.RSI(close, timeperiod=14)
        feature_columns += ["RSI"]
        return df, feature_columns

    def STOCH1(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        # STOCH - Stochastic
        df["slowk_STOCH"], df["slowd_STOCH"] = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        feature_columns += ["slowk_STOCH", "slowd_STOCH"]

        df["slowk_STOCH_slowd_STOCH_ratio"] = df["slowk_STOCH"] / df["slowd_STOCH"]
        feature_columns += ["slowk_STOCH_slowd_STOCH_ratio", "slowd_STOCH"]
        return df, feature_columns

    def STOCH2(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        # STOCHF - Stochastic Fast
        df["fastk_STOCHF"], df["fastd_STOCHF"] = talib.STOCHF(high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0)
        df["fastk_STOCHF_fastd_STOCHF_ratio"] = df["fastk_STOCHF"] / df["fastd_STOCHF"]
        feature_columns += ["fastk_STOCHF", "fastd_STOCHF", "fastk_STOCHF_fastd_STOCHF_ratio"]
        return df, feature_columns

    def STOCH3(df):
        feature_columns = []
        close = df["close"].values
        # STOCHRSI - Stochastic Relative Strength Index
        df["fastk_STOCHRSI"], df["fastd_STOCHRSI"] = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        df["fastk_STOCHRSI_fastd_STOCHRSI_ratio"] = df["fastk_STOCHRSI"] / df["fastd_STOCHRSI"]
        feature_columns += ["fastk_STOCHRSI", "fastd_STOCHRSI", "fastk_STOCHRSI_fastd_STOCHRSI_ratio"]
        return df, feature_columns

    def TRIX(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
        df["TRIX"] = talib.TRIX(close, timeperiod=30)
        feature_columns += ["TRIX"]

        # ULTOSC - Ultimate Oscillator
        df["ULTOSC"] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        feature_columns += ["ULTOSC"]

        # WILLR - Williams' %R
        df["WILLR"] = talib.WILLR(high, low, close, timeperiod=14)
        feature_columns += ["WILLR"]

        return df, feature_columns

    # ===================
    #  Volatility Indicator Functions
    # ===================

    def ATR(df):
        feature_columns = []
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        # ATR - Average True Range
        df["ATR"] = talib.ATR(high, low, close, timeperiod=14)
        # NATR - Normalized Average True Range
        df["NATR"] = talib.NATR(high, low, close, timeperiod=14)
        # TRANGE - True Range
        df["TRANGE"] = talib.TRANGE(high, low, close)

        feature_columns += ["ATR", "NATR", "TRANGE"]
        return df, feature_columns

    # ===================
    #  Cycle Indicator Functions
    # ===================
    def HT_DCPERIOD(df):
        feature_columns = []
        close = df["close"].values

        # HT_DCPERIOD - Hilbert Transform - DominMAXant Cycle Period
        df["HT_DCPERIOD"] = talib.HT_DCPERIOD(close)
        # HT_DCPHASE - Hilbert Transform - DominMAXant Cycle Phase
        df["HT_DCPHASE"] = talib.HT_DCPHASE(close)
        feature_columns += ["HT_DCPERIOD", "HT_DCPHASE"]
        return df, feature_columns

    def HT_DCPERIOD2(df):
        feature_columns = []
        close = df["close"].values
        # HT_PHASOR - Hilbert Transform - Phasor Components
        df["inphase"], df["quadrature"] = talib.HT_PHASOR(close)
        df["inphase_quadrature_ratio"] = df["inphase"] / df["quadrature"]

        feature_columns += ["inphase", "quadrature", "inphase_quadrature_ratio"]

        return df, feature_columns

    def sine(df):
        feature_columns = []
        close = df["close"].values
        # HT_SINE - Hilbert Transform - SineWave
        df["sine"], df["leadsine"] = talib.HT_SINE(close)
        df["sine_leadsine_ratio"] = df["sine"] / df["leadsine"]

        # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
        df["integer"] = talib.HT_TRENDMODE(close)

        feature_columns += ["sine", "leadsine", "sine_leadsine_ratio", "integer"]
        return df, feature_columns

    # ===================
    #  Price Transform Functions
    # ===================
    def PRICE(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        df["AVGPRICE"] = talib.AVGPRICE(open, high, low, close)
        df["MEDPRICE"] = talib.MEDPRICE(high, low)
        df["TYPPRICE"] = talib.TYPPRICE(high, low, close)
        df["WCLPRICE"] = talib.WCLPRICE(high, low, close)

        feature_columns += ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"]
        return df, feature_columns

    # ===================
    #  Pattern Recognition Functions
    # ===================

    def pattern_recognition2(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDL3STARSINSOUTH"] = talib.CDL3STARSINSOUTH(open, high, low, close)
        df["CDL3WHITESOLDIERS"] = talib.CDL3WHITESOLDIERS(open, high, low, close)
        df["CDLABANDONEDBABY"] = talib.CDLABANDONEDBABY(open, high, low, close, penetration=0)
        df["CDLADVANCEBLOCK"] = talib.CDLADVANCEBLOCK(open, high, low, close)

        feature_columns += [
            "CDL3STARSINSOUTH",
            "CDL3WHITESOLDIERS",
            "CDLABANDONEDBABY",
            "CDLADVANCEBLOCK",
        ]
        return df, feature_columns

    def pattern_recognition3(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLCOUNTERATTACK"] = talib.CDLCOUNTERATTACK(open, high, low, close)
        df["CDLDARKCLOUDCOVER"] = talib.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0)
        df["CDLDOJI"] = talib.CDLDOJI(open, high, low, close)
        df["CDLDOJISTAR"] = talib.CDLDOJISTAR(open, high, low, close)
        feature_columns += [
            "CDLCOUNTERATTACK",
            "CDLDARKCLOUDCOVER",
            "CDLDOJI",
            "CDLDOJISTAR",
        ]
        return df, feature_columns

    def pattern_recognition4(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLGAPSIDESIDEWHITE"] = talib.CDLGAPSIDESIDEWHITE(open, high, low, close)
        df["CDLGRAVESTONEDOJI"] = talib.CDLGRAVESTONEDOJI(open, high, low, close)
        df["CDLHAMMER"] = talib.CDLHAMMER(open, high, low, close)
        df["CDLHANGINGMAN"] = talib.CDLHANGINGMAN(open, high, low, close)
        feature_columns += [
            "CDLGAPSIDESIDEWHITE",
            "CDLGRAVESTONEDOJI",
            "CDLHAMMER",
            "CDLHANGINGMAN",
        ]
        return df, feature_columns

    def pattern_recognition5(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        # df["CDLHOMINMAXGPIGEON"] = talib.CDLHOMINMAXGPIGEON(open, high, low, close)
        df["CDLIDENTICAL3CROWS"] = talib.CDLIDENTICAL3CROWS(open, high, low, close)
        df["CDLINNECK"] = talib.CDLINNECK(open, high, low, close)
        df["CDLINVERTEDHAMMER"] = talib.CDLINVERTEDHAMMER(open, high, low, close)
        feature_columns += [
            # "CDLHOMINMAXGPIGEON",
            "CDLIDENTICAL3CROWS",
            "CDLINNECK",
            "CDLINVERTEDHAMMER",
        ]
        return df, feature_columns

    def pattern_recognition6(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLMARUBOZU"] = talib.CDLMARUBOZU(open, high, low, close)
        df["CDLMATCHINGLOW"] = talib.CDLMATCHINGLOW(open, high, low, close)
        df["CDLMATHOLD"] = talib.CDLMATHOLD(open, high, low, close, penetration=0)
        df["CDLMORNINGDOJISTAR"] = talib.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0)
        feature_columns += [
            "CDLMARUBOZU",
            "CDLMATCHINGLOW",
            "CDLMATHOLD",
            "CDLMORNINGDOJISTAR",
        ]
        return df, feature_columns

    def pattern_recognition7(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLRISEFALL3METHODS"] = talib.CDLRISEFALL3METHODS(open, high, low, close)
        df["CDLSEPARATINGLINES"] = talib.CDLSEPARATINGLINES(open, high, low, close)
        df["CDLSHOOTINGSTAR"] = talib.CDLSHOOTINGSTAR(open, high, low, close)
        feature_columns += [
            "CDLRISEFALL3METHODS",
            "CDLSEPARATINGLINES",
            "CDLSHOOTINGSTAR",
        ]
        return df, feature_columns

    def pattern_recognition8(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLTAKURI"] = talib.CDLTAKURI(open, high, low, close)
        df["CDLTASUKIGAP"] = talib.CDLTASUKIGAP(open, high, low, close)
        df["CDLTHRUSTING"] = talib.CDLTHRUSTING(open, high, low, close)
        feature_columns += [
            "CDLTAKURI",
            "CDLTASUKIGAP",
            "CDLTHRUSTING",
        ]
        return df, feature_columns

    def pattern_recognition9(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLBELTHOLD"] = talib.CDLBELTHOLD(open, high, low, close)
        df["CDLBREAKAWAY"] = talib.CDLBREAKAWAY(open, high, low, close)
        df["CDLCLOSINGMARUBOZU"] = talib.CDLCLOSINGMARUBOZU(open, high, low, close)
        df["CDLCONCEALBABYSWALL"] = talib.CDLCONCEALBABYSWALL(open, high, low, close)
        feature_columns += [
            "CDLBELTHOLD",
            "CDLBREAKAWAY",
            "CDLCLOSINGMARUBOZU",
            "CDLCONCEALBABYSWALL",
        ]
        return df, feature_columns

    def pattern_recognition10(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLDRAGONFLYDOJI"] = talib.CDLDRAGONFLYDOJI(open, high, low, close)
        df["CDLENGULFING"] = talib.CDLENGULFING(open, high, low, close)
        df["CDLEVENINGDOJISTAR"] = talib.CDLEVENINGDOJISTAR(open, high, low, close, penetration=0)
        df["CDLEVENINGSTAR"] = talib.CDLEVENINGSTAR(open, high, low, close, penetration=0)
        feature_columns += [
            "CDLDRAGONFLYDOJI",
            "CDLENGULFING",
            "CDLEVENINGDOJISTAR",
            "CDLEVENINGSTAR",
        ]
        return df, feature_columns

    def pattern_recognition11(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLHARAMICROSS"] = talib.CDLHARAMICROSS(open, high, low, close)
        df["CDLHIGHWAVE"] = talib.CDLHIGHWAVE(open, high, low, close)
        df["CDLHIKKAKE"] = talib.CDLHIKKAKE(open, high, low, close)
        df["CDLHIKKAKEMOD"] = talib.CDLHIKKAKEMOD(open, high, low, close)
        feature_columns += [
            "CDLHARAMICROSS",
            "CDLHIGHWAVE",
            "CDLHIKKAKE",
            "CDLHIKKAKEMOD",
        ]
        return df, feature_columns

    def pattern_recognition12(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLKICKING"] = talib.CDLKICKING(open, high, low, close)
        df["CDLKICKINGBYLENGTH"] = talib.CDLKICKINGBYLENGTH(open, high, low, close)
        # df["CDLLDIVERBOTTOM"] = talib.CDLLDIVERBOTTOM(open, high, low, close)
        df["CDLLONGLEGGEDDOJI"] = talib.CDLLONGLEGGEDDOJI(open, high, low, close)
        df["CDLLONGLINE"] = talib.CDLLONGLINE(open, high, low, close)
        feature_columns += [
            "CDLKICKING",
            "CDLKICKINGBYLENGTH",
            # "CDLLDIVERBOTTOM",
            "CDLLONGLEGGEDDOJI",
            "CDLLONGLINE",
        ]
        return df, feature_columns

    def pattern_recognition13(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLMORNINGSTAR"] = talib.CDLMORNINGSTAR(open, high, low, close, penetration=0)
        df["CDLONNECK"] = talib.CDLONNECK(open, high, low, close)
        df["CDLPIERCING"] = talib.CDLPIERCING(open, high, low, close)
        df["CDLRICKSHAWMAN"] = talib.CDLRICKSHAWMAN(open, high, low, close)
        feature_columns += [
            "CDLMORNINGSTAR",
            "CDLONNECK",
            "CDLPIERCING",
            "CDLRICKSHAWMAN",
        ]
        return df, feature_columns

    def pattern_recognition14(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDL2CROWS"] = talib.CDL2CROWS(open, high, low, close)
        df["CDL3BLACKCROWS"] = talib.CDL3BLACKCROWS(open, high, low, close)
        feature_columns += ["CDL2CROWS", "CDL3BLACKCROWS"]
        return df, feature_columns

    def pattern_recognition15(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLSHORTLINE"] = talib.CDLSHORTLINE(open, high, low, close)
        df["CDLSPINNINGTOP"] = talib.CDLSPINNINGTOP(open, high, low, close)
        df["CDLSTALLEDPATTERN"] = talib.CDLSTALLEDPATTERN(open, high, low, close)
        df["CDLSTICKSANDWICH"] = talib.CDLSTICKSANDWICH(open, high, low, close)
        feature_columns += [
            "CDLSHORTLINE",
            "CDLSPINNINGTOP",
            "CDLSTALLEDPATTERN",
            "CDLSTICKSANDWICH",
        ]
        return df, feature_columns

    def pattern_recognition16(df):
        feature_columns = []
        open = df["open"].values
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        df["CDLTRISTAR"] = talib.CDLTRISTAR(open, high, low, close)
        df["CDLUNIQUE3RIVER"] = talib.CDLUNIQUE3RIVER(open, high, low, close)
        df["CDLUPSIDEGAP2CROWS"] = talib.CDLUPSIDEGAP2CROWS(open, high, low, close)
        df["CDLXSIDEGAP3METHODS"] = talib.CDLXSIDEGAP3METHODS(open, high, low, close)
        feature_columns += [
            "CDLTRISTAR",
            "CDLUNIQUE3RIVER",
            "CDLUPSIDEGAP2CROWS",
            "CDLXSIDEGAP3METHODS",
        ]
        return df, feature_columns

    # ===================
    #  Statistic Functions
    # ===================
    def BETA(df):
        # BETA - Beta
        high = df["high"].values
        low = df["low"].values
        for i in [5, 30, 14, 120]:
            feature_columns = []
            df[f"BETA{i}"] = talib.BETA(high, low, timeperiod=i)
            feature_columns += [f"BETA{i}"]
            return df, feature_columns

    def CORREL(df):
        high = df["high"].values
        low = df["low"].values
        for i in [5, 30, 14, 120]:
            feature_columns = []
            df[f"CORREL{i}"] = talib.CORREL(high, low, timeperiod=i)
            feature_columns += [f"CORREL{i}"]
            return df, feature_columns

    def LINEARREG(df):
        close = df["close"].values
        for i in [5, 30, 14, 120]:
            feature_columns = []
            df[f"LINEARREG{i}"] = talib.LINEARREG(close, timeperiod=i)
            feature_columns += [f"LINEARREG{i}"]
            return df, feature_columns

    def LINEARREG_ANGLE(df):
        close = df["close"].values
        for i in [5, 30, 14, 120]:
            feature_columns = []
            df[f"LINEARREG_ANGLE{i}"] = talib.LINEARREG_ANGLE(close, timeperiod=i)
            feature_columns += [f"LINEARREG_ANGLE{i}"]
            return df, feature_columns

    def LINEARREG_SLOPE(df):
        close = df["close"].values
        for i in [5, 30, 14, 120]:
            feature_columns = []
            df[f"LINEARREG_SLOPE{i}"] = talib.LINEARREG_SLOPE(close, timeperiod=i)
            feature_columns += [f"LINEARREG_SLOPE{i}"]
            return df, feature_columns

    def STDDEV(df):
        close = df["close"].values
        feature_columns = []
        df["STDDEV"] = talib.STDDEV(close, timeperiod=5, nbdev=1)
        feature_columns += ["STDDEV"]
        return df, feature_columns

    def TSF(df):
        close = df["close"].values
        for i in [5, 30, 14, 120]:
            feature_columns = []
            df[f"TSF{i}"] = talib.TSF(close, timeperiod=i)
            feature_columns += [f"TSF{i}"]
            return df, feature_columns

    def VAR(df):
        close = df["close"].values
        feature_columns = []
        df["VAR"] = talib.VAR(close, timeperiod=5, nbdev=1)
        feature_columns += ["VAR"]
        return df, feature_columns

    # ===================
    #  Math Transform Functions
    # ===================
    def ACOS(df):
        close = df["close"].values
        feature_columns = []
        df["ACOS"] = talib.ACOS(close)
        feature_columns += ["ACOS"]
        return df, feature_columns

    def TANH(df):
        close = df["close"].values
        feature_columns = []
        df["TANH"] = talib.TANH(close)
        feature_columns += ["TANH"]
        return df, feature_columns

    def TAN(df):
        close = df["close"].values
        feature_columns = []
        df["TAN"] = talib.TAN(close)
        feature_columns += ["TAN"]
        return df, feature_columns

    def SINH(df):
        close = df["close"].values
        feature_columns = []
        df["SINH"] = talib.SINH(close)
        feature_columns += ["SINH"]
        return df, feature_columns

    def SIN(df):
        close = df["close"].values
        feature_columns = []
        df["SIN"] = talib.SIN(close)
        feature_columns += ["SIN"]
        return df, feature_columns

    def LOG10(df):
        close = df["close"].values
        feature_columns = []
        df["LOG10"] = talib.LOG10(close)
        feature_columns += ["LOG10"]
        return df, feature_columns

    def LN(df):
        close = df["close"].values
        feature_columns = []
        df["LN"] = talib.LN(close)
        feature_columns += ["LN"]
        return df, feature_columns

    def FLOOR(df):
        close = df["close"].values
        feature_columns = []
        df["FLOOR"] = talib.FLOOR(close)
        feature_columns += ["FLOOR"]
        return df, feature_columns

    def EXP(df):
        close = df["close"].values
        feature_columns = []
        df["EXP"] = talib.EXP(close)
        feature_columns += ["EXP"]
        return df, feature_columns

    def COSH(df):
        close = df["close"].values
        feature_columns = []
        df["COSH"] = talib.COSH(close)
        feature_columns += ["COSH"]
        return df, feature_columns

    def COS(df):
        close = df["close"].values
        feature_columns = []
        df["COS"] = talib.COS(close)
        feature_columns += ["COS"]
        return df, feature_columns

    def CEIL(df):
        close = df["close"].values
        feature_columns = []
        df["CEIL"] = talib.CEIL(close)
        feature_columns += ["CEIL"]
        return df, feature_columns

    def ATAN(df):
        close = df["close"].values
        feature_columns = []
        df["ATAN"] = talib.ATAN(close)
        feature_columns += ["ATAN"]
        return df, feature_columns

    def ASIN(df):
        close = df["close"].values
        feature_columns = []
        df["ASIN"] = talib.ASIN(close)
        feature_columns += ["ASIN"]
        return df, feature_columns

    # ===================
    #  Math Operator Functions
    # ===================
    def MINMAX(df):
        close = df["close"].values
        feature_columns = []
        for i in [5, 30, 60, 120]:
            df[f"MINMAX{i}"] = talib.MINMAX(close, timeperiod=i)
            feature_columns += [f"MINMAX{i}"]
        return df, feature_columns

    def MIN(df):
        close = df["close"].values
        feature_columns = []
        for i in [5, 30, 60, 120]:
            df[f"MIN{i}"] = talib.MIN(close, timeperiod=i)
            feature_columns += [f"MIN{i}"]
        return df, feature_columns

    def MAX(df):
        close = df["close"].values
        feature_columns = []
        for i in [5, 30, 60, 120]:
            df[f"MAX{i}"] = talib.MAX(close, timeperiod=i)
            feature_columns += [f"MAX{i}"]
        return df, feature_columns

    def DIV(df):
        high = df["high"].values
        low = df["low"].values
        feature_columns = []
        df["DIV"] = talib.DIV(high, low)
        feature_columns += ["DIV"]
        return df, feature_columns

    def MULT(df):
        high = df["high"].values
        low = df["low"].values
        feature_columns = []
        df["MULT"] = talib.MULT(high, low)
        feature_columns += ["MULT"]
        return df, feature_columns

    def SUB(df):
        high = df["high"].values
        low = df["low"].values
        feature_columns = []
        df["SUB"] = talib.SUB(high, low)
        feature_columns += ["SUB"]
        return df, feature_columns

    # def SUM(df):
    #     high = df["high"].values
    #     low = df["low"].values
    #     feature_columns = []
    #     df["SUM"] = talib.SUM(high, low)
    #     feature_columns += ["SUM"]
    #     return df, feature_columns

    def ADD(df):
        high = df["high"].values
        low = df["low"].values
        feature_columns = []
        df["ADD"] = talib.ADD(high, low)
        feature_columns += ["ADD"]
        return df, feature_columns
