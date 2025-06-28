import numpy as np
import pandas as pd
import talib as ta
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
                "email": "",
                "password": ""
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
            return None
        except Exception as e:
            logger.error(f"خطأ في حساب المؤشر {name}: {str(e)}")
            return None
            
    def _calculate_classic_indicator(self, name: str, data: pd.DataFrame, settings: dict) -> Tuple[str, pd.Series]:
        """حساب المؤشرات الكلاسيكية"""
        if name == 'RSI':
            return name, ta.RSI(data['Close'], timeperiod=settings.get('period', 14))
        elif name == 'MACD':
            macd, signal, _ = ta.MACD(data['Close'])
            return name, macd - signal
        elif name == 'BB':
            upper, middle, lower = ta.BBANDS(data['Close'], timeperiod=settings.get('period', 20))
            return name, (data['Close'] - middle) / (upper - lower)
            
    # تحديث المؤشرات الموجودة مع إضافة المزيد من التحسينات والتوثيق
    def calculate_cti(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """مؤشر التراكمي الذكي المحسن للتحركات الصغيرة"""
        direction = np.where(data['Close'].diff() > 0, 1, -1)
        magnitude = np.log(data['Close'].diff().abs() / data['Close'].shift(1) + 1)
        volatility_adj = data['Close'].pct_change().rolling(period).std()
        cti = (direction * magnitude * (1 + volatility_adj)).ewm(span=period).mean()
        return cti * 100

    # ... (إضافة باقي المؤشرات المحسنة)

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
            
        try:
            proba = self.model.predict_proba(X)[:, 1]
            confidence = abs(proba[-1] - 0.5) * 2  # تحويل الاحتمالية إلى درجة ثقة
            
            if proba[-1] > self.config['threshold']:
                return 1, confidence  # إشارة شراء
            elif proba[-1] < (1 - self.config['threshold']):
                return -1, confidence  # إشارة بيع
            return 0, confidence  # حياد
            
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
        joblib.dump(self.model, self.model_path)
        
    def _load_model(self) -> None:
        """تحميل النموذج المحفوظ"""
        try:
            self.model = joblib.load(self.model_path)
        except:
            logger.warning("لم يتم العثور على نموذج محفوظ")
            
    # Helper methods for feature engineering
    def _calculate_price_momentum(self, data: pd.DataFrame) -> pd.Series:
        """حساب زخم السعر"""
        return data['Close'].pct_change(5).rolling(10).mean()
        
    def _calculate_volume_force(self, data: pd.DataFrame) -> pd.Series:
        """حساب قوة الحجم"""
        return (data['Volume'] * data['Close'].pct_change()).rolling(5).sum()
        
    def _detect_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """تحديد نظام السوق"""
        volatility = data['Close'].pct_change().rolling(20).std()
        trend = data['Close'].pct_change(20)
        return pd.qcut(volatility * abs(trend), q=3, labels=[-1, 0, 1], duplicates='drop')
        
    def _prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """تحضير المتغير التابع"""
        future_returns = data['Close'].shift(-self.config['lookback_period']).pct_change(self.config['lookback_period'])
        return (future_returns > 0).astype(int)
        
    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """تنظيف البيانات"""
        valid_idx = ~X.isna().any(axis=1) & ~y.isna()
        return X[valid_idx], y[valid_idx]

class NotificationManager:
    """إدارة الإشعارات والتنبيهات"""
    
    def __init__(self, config: dict):
        self.config = config['email_notifications']
        
    def send_alert(self, symbol: str, signal: int, confidence: float) -> bool:
        """إرسال تنبيه عبر البريد الإلكتروني"""
        if not self.config['enabled']:
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
                server.starttls()
                server.login(self.config['email'], self.config['password'])
                server.send_message(msg)
                
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إرسال التنبيه: {str(e)}")
            return False

class EnhancedDashboard:
    """لوحة التحكم المحسنة"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.data_manager = DataManager()
        self.indicators = EnhancedIndicators(self.config_manager.config)
        self.ai_model = EnhancedAIModel(self.config_manager.config)
        self.notification_manager = NotificationManager(self.config_manager.config)
        
    def show_dashboard(self):
        """عرض لوحة التحكم المحسنة"""
        st.set_page_config(
            layout="wide",
            page_title="نظام التداول الذكي المتقدم",
            page_icon="📈"
        )
        
        self._setup_styles()
        self._show_header()
        
        # تقسيم الشاشة
        col1, col2 = st.columns([1, 3])
        
        with col1:
            self._show_sidebar_controls()
            
        with col2:
            self._show_main_content()
            
    def _setup_styles(self):
        """إعداد الأنماط"""
        st.markdown("""
        <style>
        .stSlider > div { padding: 0 10px; }
        .stTextInput > div > div > input { direction: ltr; text-align: right; }
        .sidebar .sidebar-content { direction: rtl; }
        .big-font { font-size: 24px !important; }
        .highlight { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
        </style>
        """, unsafe_allow_html=True)
        
    def _show_header(self):
        """عرض رأس الصفحة"""
        st.title("📊 نظام التداول الذكي المتقدم")
        st.markdown("---")
        
    def _show_sidebar_controls(self):
        """عرض أدوات التحكم في الشريط الجانبي"""
        st.sidebar.header("⚙️ إعدادات النظام")
        
        # اختيار الأداة المالية
        symbol = st.sidebar.selectbox(
            "اختر الأداة المالية",
            self.config_manager.config['trading_pairs']
        )
        
        # إضافة أداة مالية جديدة
        new_symbol = st.sidebar.text_input("إضافة أداة مالية جديدة")
        if new_symbol:
            if new_symbol not in self.config_manager.config['trading_pairs']:
                self.config_manager.config['trading_pairs'].append(new_symbol)
                self.config_manager.save_config(self.config_manager.config)
                st.sidebar.success("تمت إضافة الأداة المالية بنجاح")
                
        # اختيار الفترة الزمنية
        period = st.sidebar.radio(
            "الفترة الزمنية",
            ['1w', '1m', '3m', '6m', '1y', '2y'],
            index=2
        )
        
        # إعدادات المؤشرات
        st.sidebar.subheader("ضبط المؤشرات الفنية")
        for name, config in self.config_manager.config['indicators'].items():
            col1, col2, col3 = st.sidebar.columns([1, 2, 1])
            with col1:
                config['active'] = st.checkbox(
                    f"{name}",
                    value=config['active'],
                    key=f"{name}_active"
                )
            with col2:
                if 'period' in config:
                    config['period'] = st.slider(
                        "الفترة",
                        5, 100, config['period'],
                        key=f"{name}_period"
                    )
            with col3:
                config['weight'] = st.slider(
                    "الوزن",
                    0.0, 1.0, config['weight'],
                    key=f"{name}_weight"
                )
                
        # حفظ التغييرات
        if st.sidebar.button("💾 حفظ الإعدادات"):
            self.config_manager.save_config(self.config_manager.config)
            st.sidebar.success("تم حفظ الإعدادات بنجاح")
            
        return symbol, period
        
    def _show_main_content(self):
        """عرض المحتوى الرئيسي"""
        symbol, period = self._show_sidebar_controls()
        
        if st.button("🔄 تحديث التحليل"):
            self._run_analysis(symbol, period)
            
    def _run_analysis(self, symbol: str, period: str):
        """تشغيل التحليل"""
        with st.spinner("جاري تحليل السوق وتوليد الإشارات..."):
            try:
                # تحميل البيانات
                data = self.data_manager.get_data(symbol, period)
                
                # حساب المؤشرات
                indicators_data = self.indicators.calculate_all_indicators(data)
                
                # تدريب النموذج
                X, y = self.ai_model.prepare_data(data, indicators_data)
                metrics = self.ai_model.train_model(X, y)
                
                # توليد الإشارة
                signal, confidence = self.ai_model.predict_signal(X.iloc[-1:])
                
                # عرض النتائج
                self._show_results(data, indicators_data, metrics, signal, confidence)
                
                # إرسال تنبيه
                if abs(signal) == 1 and confidence > 0.8:
                    self.notification_manager.send_alert(symbol, signal, confidence)
                    
            except Exception as e:
                st.error(f"حدث خطأ أثناء التحليل: {str(e)}")
                logger.error(f"خطأ في التحليل: {str(e)}")
                
    def _show_results(self, data: pd.DataFrame, indicators: pd.DataFrame,
                     metrics: dict, signal: int, confidence: float):
        """عرض نتائج التحليل"""
        # عرض المقاييس الرئيسية
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._metric_card("دقة النموذج", f"{metrics['accuracy']:.2%}")
        with col2:
            self._metric_card("F1 Score", f"{metrics['f1']:.2%}")
        with col3:
            signal_text = "شراء 🟢" if signal == 1 else "بيع 🔴" if signal == -1 else "حياد ⚪"
            self._metric_card("الإشارة", signal_text)
        with col4:
            self._metric_card("درجة الثقة", f"{confidence:.2%}")
            
        # عرض التحليل البياني
        tab1, tab2, tab3 = st.tabs(["الرسم البياني", "المؤشرات الفنية", "تحليل النموذج"])
        
        with tab1:
            self._plot_price_chart(data, signal)
            
        with tab2:
            self._plot_indicators(data, indicators)
            
        with tab3:
            self._show_model_analysis()
            
    def _metric_card(self, title: str, value: str):
        """عرض بطاقة مقياس"""
        st.markdown(f"""
        <div class="highlight">
            <p style="font-size: 14px; color: gray;">{title}</p>
            <p class="big-font">{value}</p>
        </div>
        """, unsafe_allow_html=True)
        
    def _plot_price_chart(self, data: pd.DataFrame, signal: int):
        """رسم المخطط السعري"""
        fig = make_subplots(rows=2, cols=1, shared_xaxis=True,
                          vertical_spacing=0.03,
                          row_heights=[0.7, 0.3])
                          
        # إضافة شمعة الأسعار
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="الأسعار"
            ),
            row=1, col=1
        )
        
        # إضافة حجم التداول
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="الحجم",
                marker_color='rgb(158,202,225)'
            ),
            row=2, col=1
        )
        
        # إضافة إشارة التداول
        if abs(signal) == 1:
            last_price = data['Close'].iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[data.index[-1]],
                    y=[last_price],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if signal == 1 else 'triangle-down',
                        size=15,
                        color='green' if signal == 1 else 'red'
                    ),
                    name="إشارة التداول"
                ),
                row=1, col=1
            )
            
        # تخصيص المخطط
        fig.update_layout(
            title="تحليل السعر والحجم",
            xaxis_title="التاريخ",
            yaxis_title="السعر",
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _plot_indicators(self, data: pd.DataFrame, indicators: pd.DataFrame):
        """رسم المؤشرات الفنية"""
        active_indicators = [name for name, config in 
                           self.config_manager.config['indicators'].items()
                           if config['active']]
                           
        n_indicators = len(active_indicators)
        if n_indicators == 0:
            st.warning("لا توجد مؤشرات نشطة للعرض")
            return
            
        # إنشاء مخطط لكل مؤشر
        fig = make_subplots(
            rows=n_indicators,
            cols=1,
            shared_xaxis=True,
            vertical_spacing=0.03,
            subplot_titles=active_indicators
        )
        
        for i, indicator_name in enumerate(active_indicators, 1):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators[indicator_name],
                    name=indicator_name,
                    line=dict(width=1)
                ),
                row=i, col=1
            )
            
        fig.update_layout(
            height=200 * n_indicators,
            showlegend=True,
            title_text="المؤشرات الفنية"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _show_model_analysis(self):
        """عرض تحليل النموذج"""
        if self.ai_model.feature_importances:
            st.subheader("أهمية المؤشرات في النموذج")
            
            # تحويل أهمية المؤشرات إلى DataFrame
            fi_df = pd.DataFrame.from_dict(
                self.ai_model.feature_importances,
                orient='index',
                columns=['الأهمية']
            ).sort_values('الأهمية', ascending=True)
            
            # رسم مخطط شريطي أفقي
            fig = go.Figure(
                go.Bar(
                    y=fi_df.index,
                    x=fi_df['الأهمية'],
                    orientation='h'
                )
            )
            
            fig.update_layout(
                title="أهمية المؤشرات",
                xaxis_title="الأهمية النسبية",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        # عرض مقاييس الأداء
        st.subheader("مقاييس أداء النموذج")
        metrics_df = pd.DataFrame({
            'المقياس': ['الدقة', 'F1 Score', 'ROC AUC'],
            'القيمة': [
                f"{self.ai_model.metrics['accuracy']:.2%}",
                f"{self.ai_model.metrics['f1']:.2%}",
                f"{self.ai_model.metrics['roc_auc']:.2%}"
            ]
        })
        
        st.table(metrics_df)

if __name__ == "__main__":
    dashboard = EnhancedDashboard()
    dashboard.show_dashboard()