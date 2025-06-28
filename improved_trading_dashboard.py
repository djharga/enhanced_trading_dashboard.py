import numpy as np
import pandas as pd
import pandas_ta as ta  # تم التعديل: استخدام pandas_ta بدلاً من talib
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb
import streamlit as st
from datetime import datetime, timedelta
import warnings
import joblib
import json
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
import logging
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """إدارة الإعدادات والتكوينات"""
    
    def __init__(self):
        self.config_file = "trading_config.json"
        self.default_config = {
            "email_notifications": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "email": "your_email@gmail.com", # استبدل ببريدك الإلكتروني
                "password": "your_app_password" # استبدل بكلمة مرور التطبيق الخاصة بك
            },
            "trading_pairs": [
                "BTC-USD", "ETH-USD", "AAPL", "TSLA", "EURUSD=X"
            ],
            "indicators": {
                "CTI": {"period": 14, "active": True, "weight": 0.15},
                "VPIN": {"period": 20, "active": True, "weight": 0.12},
                "AMV": {"period": 14, "active": True, "weight": 0.18},
                "TSD": {"period": 10, "active": True, "weight": 0.15},
                "QMS": {"period": 5, "active": True, "weight": 0.2},
                "NVI": {"period": 255, "active": True, "weight": 0.1},
                "PFE": {"period": 14, "active": True, "weight": 0.1},
                # إضافة المؤشرات الكلاسيكية
                "RSI": {"period": 14, "active": True, "weight": 0.1},
                "MACD": {"active": True, "weight": 0.1},
                "BB": {"period": 20, "active": True, "weight": 0.1}
            },
            "model": {
                "lookback_period": 5,
                "test_size": 0.2,
                "threshold": 0.8,
                "hyperparameters": {
                    "n_estimators": 150,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8
                }
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """تحميل الإعدادات من الملف"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.save_config(self.default_config)
            return self.default_config
    
    def save_config(self, config: dict) -> None:
        """حفظ الإعدادات في الملف"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
            
class DataManager:
    """إدارة البيانات وتحميلها ومعالجتها"""
    
    def __init__(self):
        self.cache = {}
        
    def get_data(self, symbol: str, period: str = '1y', force_reload: bool = False) -> pd.DataFrame:
        """جلب البيانات مع التخزين المؤقت"""
        cache_key = f"{symbol}_{period}"
        
        if not force_reload and cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            end_date = datetime.now()
            start_date = self._calculate_start_date(period, end_date)
            
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"لا توجد بيانات متاحة للرمز {symbol}")
                
            # معالجة القيم المفقودة
            data = self._handle_missing_values(data)
            
            self.cache[cache_key] = data
            return data
            
        except Exception as e:
            logger.error(f"خطأ في تحميل البيانات للرمز {symbol}: {str(e)}")
            raise
            
    def _calculate_start_date(self, period: str, end_date: datetime) -> datetime:
        """حساب تاريخ البداية بناءً على الفترة"""
        period_map = {
            '1w': timedelta(days=7),
            '1m': timedelta(days=30),
            '3m': timedelta(days=90),
            '6m': timedelta(days=180),
            '1y': timedelta(days=365),
            '2y': timedelta(days=730)
        }
        return end_date - period_map.get(period, timedelta(days=365))
        
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """معالجة القيم المفقودة في البيانات"""
        # ملء القيم المفقودة باستخدام الطريقة المناسبة
        data = data.fillna(method='ffill')
        if data.isnull().any().any():
            data = data.fillna(method='bfill')
        return data

class EnhancedIndicators:
    """المؤشرات الفنية المحسنة"""
    
    def __init__(self, config: dict):
        self.config = config['indicators']
        
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """حساب جميع المؤشرات النشطة"""
        results = pd.DataFrame(index=data.index)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for name, settings in self.config.items():
                if settings['active']:
                    futures.append(
                        executor.submit(
                            self._calculate_indicator,
                            name,
                            data,
                            settings
                        )
                    )
                    
            for future in futures:
                indicator_data = future.result()
                if indicator_data is not None:
                    name, values = indicator_data
                    results[name] = values
                    
        return results
        
    def _calculate_indicator(self, name: str, data: pd.DataFrame, settings: dict) -> Tuple[str, pd.Series]:
        """حساب مؤشر فني محدد"""
        try:
            if name in ['RSI', 'MACD', 'BB']:
                return self._calculate_classic_indicator(name, data, settings)
            else:
                method = getattr(self, f"calculate_{name.lower()}", None)
                if method:
                    return name, method(data, settings.get('period', 14))
            return None, None
        except Exception as e:
            logger.error(f"خطأ في حساب المؤشر {name}: {str(e)}")
            return None, None
            
    def _calculate_classic_indicator(self, name: str, data: pd.DataFrame, settings: dict) -> Tuple[str, pd.Series]:
        """حساب المؤشرات الكلاسيكية باستخدام pandas_ta"""
        if name == 'RSI':
            # pandas_ta يضيف المؤشر كعمود جديد مباشرة
            rsi_series = data.ta.rsi(length=settings.get('period', 14))
            return name, rsi_series
        elif name == 'MACD':
            # pandas_ta يعيد DataFrame بأعمدة MACD, Histogram, Signal
            macd_df = data.ta.macd(append=False) # append=False لتجنب الإضافة المباشرة للـ DataFrame
            # نستخدم الفرق بين MACD و Signal Line كقيمة واحدة للمؤشر
            # تأكد من أسماء الأعمدة التي ينتجها pandas_ta, عادة ما تكون مثل 'MACD_12_26_9', 'MACDs_12_26_9'
            # سنقوم بحساب الفرق هنا ونستخدم آخر قيمتين فقط لإظهار الفروقات.
            if macd_df is not None and not macd_df.empty:
                # أسماء الأعمدة الافتراضية لـ pandas_ta MACD
                macd_col = f"MACD_12_26_9"
                signal_col = f"MACDs_12_26_9"
                if macd_col in macd_df.columns and signal_col in macd_df.columns:
                    return name, macd_df[macd_col] - macd_df[signal_col]
            return name, pd.Series(np.nan, index=data.index)
        elif name == 'BB':
            # pandas_ta يعيد DataFrame بـ Lower, Middle, Upper Band وأعمدة أخرى
            bbands_df = data.ta.bbands(length=settings.get('period', 20), append=False)
            # نحسب الانحراف المعياري للسعر نسبة إلى النطاق أو نستخدم نسبة%B
            # نسبة %B هي مؤشر جيد لموضع السعر داخل نطاقات بولينجر
            if bbands_df is not None and not bbands_df.empty:
                # أسماء الأعمدة الافتراضية لـ pandas_ta BBANDS
                percent_b_col = f"BBP_{settings.get('period', 20)}_2.0" # نسبة %B
                if percent_b_col in bbands_df.columns:
                    return name, bbands_df[percent_b_col]
            return name, pd.Series(np.nan, index=data.index)
        return name, pd.Series(np.nan, index=data.index) # في حالة عدم تطابق الاسم أو فشل الحساب

    # مؤشراتك المخصصة - تأكد من وجود تعريف لكل منها
    def calculate_cti(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """مؤشر التراكمي الذكي المحسن للتحركات الصغيرة"""
        direction = np.where(data['Close'].diff() > 0, 1, -1)
        magnitude = np.log(data['Close'].diff().abs() / data['Close'].shift(1) + 1)
        volatility_adj = data['Close'].pct_change().rolling(period).std()
        cti = (direction * magnitude * (1 + volatility_adj)).ewm(span=period).mean()
        return cti * 100
        
    def calculate_vpin(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """مؤشر VPIN المتطور مع تصحيح الانحراف"""
        buy_vol = np.where(data['Close'] > data['Open'], data['Volume'], 0)
        sell_vol = np.where(data['Close'] < data['Open'], data['Volume'], 0)
        vol_diff = pd.Series(sell_vol).rolling(period).sum() - pd.Series(buy_vol).rolling(period).sum()
        total_vol = data['Volume'].rolling(period).sum().replace(0, 1) # تجنب القسمة على صفر
        vpin = vol_diff / total_vol
        return vpin * 100

    def calculate_amv(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """مؤشر الدوامة الذكية التكيفية"""
        hl_range = data['High'] - data['Low']
        # استخدام ATR من pandas_ta بدلاً من TRANGE
        tr = data.ta.atr(length=1, append=False) # ATR مع فترة 1 هو نفسه TRANGE تقريباً
        amv = hl_range.rolling(period).std() / (tr.rolling(period).mean() + 1e-10) # إضافة ثابت صغير لتجنب القسمة على صفر
        return amv * 100

    def calculate_tsd(self, data: pd.DataFrame, period: int = 10) -> pd.Series:
        """التباعد الطيفي الذكي ثلاثي الأبعاد"""
        ma1 = data['Close'].ewm(span=period).mean()
        ma2 = data['Close'].ewm(span=period*2).mean()
        ma3 = data['Close'].ewm(span=period*4).mean()
        tsd = (ma1 - ma2).abs() + (ma2 - ma3).abs() + (ma1 - ma3).abs()
        return tsd / data['Close'] * 100

    def calculate_qms(self, data: pd.DataFrame, period: int = 5) -> pd.Series:
        """مؤشر الزخم الكمي متعدد الأبعاد"""
        log_ret = np.log(data['Close']/data['Close'].shift(1))
        # Ensure 'apply' is used correctly with lambda for rolling window
        qms = log_ret.rolling(period).apply(lambda x: np.sqrt(np.sum(x**2)), raw=False)
        return qms * 100

    def calculate_nvi(self, data: pd.DataFrame, period: int = 255) -> pd.Series:
        """مؤشر الحجم السلبي الذكي"""
        price_change = data['Close'].pct_change()
        nvi = pd.Series(1, index=data.index)
        for i in range(1, len(data)):
            if data['Volume'].iloc[i] < data['Volume'].iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + price_change.iloc[i])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        return nvi.rolling(period).mean()

    def calculate_pfe(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """كفاءة الفركتال القطبية"""
        # إضافة 1e-10 لتجنب القسمة على صفر في المقام
        pfe = (data['Close'] - data['Close'].shift(period)) / \
              (np.sqrt((data['Close'].diff()**2 + 1e-10).rolling(period).sum()))
        return pfe * 100


class EnhancedAIModel:
    """نموذج الذكاء الاصطناعي المحسن"""
    
    def __init__(self, config: dict):
        self.config = config['model']
        self.model = None
        self.scaler = RobustScaler()
        self.metrics = {}
        self.feature_importances = {}
        self.model_path = "trained_model.joblib"
        
    def prepare_data(self, data: pd.DataFrame, indicators: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """تحضير البيانات للتدريب"""
        X = indicators.copy()
        
        # إضافة ميزات متقدمة
        X['Price_Momentum'] = self._calculate_price_momentum(data)
        X['Volume_Force'] = self._calculate_volume_force(data)
        X['Market_Regime'] = self._detect_market_regime(data)
        
        # تحضير المتغير التابع
        y = self._prepare_target(data)
        
        # تنظيف البيانات
        return self._clean_data(X, y)
        
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """تدريب النموذج مع التحقق المتقاطع"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['test_size'],
                shuffle=False
            )
            
            model = self._create_model()
            model.fit(X_train, y_train)
            
            # حساب المقاييس
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            # حساب أهمية الميزات
            if hasattr(model.named_steps['lgbmclassifier'], 'feature_importances_'):
                self.feature_importances = dict(zip(
                    X.columns,
                    model.named_steps['lgbmclassifier'].feature_importances_
                ))
            
            self.model = model
            self._save_model()
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"خطأ في تدريب النموذج: {str(e)}")
            raise
            
    def predict_signal(self, X: pd.DataFrame) -> Tuple[int, float]:
        """توليد إشارة تداول مع درجة الثقة"""
        if self.model is None:
            self._load_model()
            # إذا لم يتم تحميل النموذج بعد المحاولة، فهذا يعني أنه غير موجود، لا يمكن التنبؤ
            if self.model is None:
                logger.error("النموذج غير موجود. لا يمكن توليد إشارة.")
                return 0, 0.0

        try:
            # التأكد أن X تحتوي على نفس الأعمدة المستخدمة في التدريب
            # وإلا سيحدث خطأ عند التحجيم أو التنبؤ
            if self.model is not None and isinstance(self.model, make_pipeline):
                # إذا كان النموذج مدرباً، استخدم الأعمدة التي تدرب عليها
                # وإلا، فافترض أن X جاهزة للتنبؤ
                
                # إذا كان scaler موجوداً في الـ pipeline، يجب أن يتعامل مع الأعمدة بشكل صحيح
                # أو يجب التأكد من تطابق الأعمدة بين X الحالية و X التي تدرب عليها النموذج
                
                # هنا نفترض أن X (current_features) تحتوي على الأعمدة الصحيحة
                proba = self.model.predict_proba(X)[:, 1]
                confidence = abs(proba[-1] - 0.5) * 2  # تحويل الاحتمالية إلى درجة ثقة
                
                if proba[-1] > self.config['threshold']:
                    return 1, confidence  # إشارة شراء
                elif proba[-1] < (1 - self.config['threshold']):
                    return -1, confidence  # إشارة بيع
                return 0, confidence  # حياد
            else:
                logger.warning("النموذج غير مدرب أو غير صحيح. لا يمكن توليد إشارة.")
                return 0, 0.0
            
        except Exception as e:
            logger.error(f"خطأ في توليد الإشارة: {str(e)}")
            return 0, 0.0
            
    def _create_model(self) -> make_pipeline:
        """إنشاء نموذج مع الإعدادات المحددة"""
        return make_pipeline(
            self.scaler,
            lgb.LGBMClassifier(
                **self.config['hyperparameters'],
                random_state=42,
                verbosity=-1
            )
        )
        
    def _save_model(self) -> None:
        """حفظ النموذج المدرب"""
        try:
            joblib.dump(self.model, self.model_path)
        except Exception as e:
            logger.error(f"خطأ في حفظ النموذج: {str(e)}")
        
    def _load_model(self) -> None:
        """تحميل النموذج المحفوظ"""
        try:
            self.model = joblib.load(self.model_path)
            logger.info("تم تحميل النموذج المحفوظ بنجاح.")
        except FileNotFoundError:
            logger.warning("لم يتم العثور على نموذج محفوظ في المسار: %s", self.model_path)
        except Exception as e:
            logger.error(f"خطأ في تحميل النموذج: {str(e)}")
            
    # Helper methods for feature engineering
    def _calculate_price_momentum(self, data: pd.DataFrame) -> pd.Series:
        """حساب زخم السعر"""
        return data['Close'].pct_change(5).rolling(10).mean()
        
    def _calculate_volume_force(self, data: pd.DataFrame) -> pd.Series:
        """حساب قوة الحجم"""
        # تأكد من أن data['Close'].pct_change() لا تحتوي على NaN في البداية لعملية الضرب
        # يمكن ملء NaN بـ 0 أو ffill
        pct_change = data['Close'].pct_change().fillna(0) 
        return (data['Volume'] * pct_change).rolling(5).sum()
        
    def _detect_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """تحديد نظام السوق"""
        volatility = data['Close'].pct_change().rolling(20).std()
        trend = data['Close'].pct_change(20)
        #fillna(0) لتجنب NaN قبل qcut
        product = (volatility * abs(trend)).fillna(0) 
        # تأكد أن هناك تنوع كافي في القيم لإنشاء 3 quantiles
        if len(product.unique()) < 3: # إذا كانت القيم كلها متطابقة تقريباً
            return pd.Series(0, index=data.index) # إرجاع قيمة افتراضية
        return pd.qcut(product, q=3, labels=[-1, 0, 1], duplicates='drop')
        
    def _prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """تحضير المتغير التابع"""
        future_returns = data['Close'].shift(-self.config['lookback_period']).pct_change(self.config['lookback_period'])
        return (future_returns > 0).astype(int)
        
    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """تنظيف البيانات"""
        # تنظيف X و y بشكل متزامن
        combined = pd.concat([X, y.rename('target')], axis=1).dropna()
        X_cleaned = combined.drop('target', axis=1)
        y_cleaned = combined['target']
        return X_cleaned, y_cleaned

class NotificationManager:
    """إدارة الإشعارات والتنبيهات"""
    
    def __init__(self, config: dict):
        self.config = config['email_notifications']
        
    def send_alert(self, symbol: str, signal: int, confidence: float) -> bool:
        """إرسال تنبيه عبر البريد الإلكتروني"""
        if not self.config['enabled']:
            logger.info("إشعارات البريد الإلكتروني غير مفعلة.")
            return False
            
        signal_map = {1: "شراء", -1: "بيع", 0: "حياد"}
        subject = f"تنبيه تداول: {signal_map[signal]} {symbol}"
        body = f"""
        تم اكتشاف إشارة {signal_map[signal]} لـ {symbol}
        درجة الثقة: {confidence:.2%}
        الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.config['email']
            msg['To'] = self.config['email']
            
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
