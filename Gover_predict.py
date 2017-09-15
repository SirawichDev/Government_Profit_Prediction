import pandas as about_t
import numpy as np
import matplotlib.pyplot as mt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")



TH_profit =[ 422895.4,
             474547.3,
             534195.09,
             572939.0,
             676988.07,
             789413.200,
             886123.990,
             843424.080,
             726394.200,
             711441.879,
             749656.077,
             792631.208,
             896814.689,
             1060754.58,
             1169622.363,
             1320594.831,
             1432913.648,
             1526804.373,
             1563011.22,
             1536592.785,
             1816746.022,
             1965116.455,
             2169854.153,
             2249957.73,
             2174749.757,
             2386463.61,
             2457860.173 ]

#เอาข้อมูลมา plotเป็นกราฟเทียบกับแต่ละปี
year_set = about_t.date_range('1990','2017',freq='A')
year_list = year_set.tolist()
#mt.plot(year_list,TH_profit)
#mt.show()
Dframe = about_t.DataFrame({'Date': year_set.values,'Value':TH_profit})
Dframe['Date_ordinal'] = Dframe['Date'].apply(lambda x:x.toordinal())

#ลองpredict10ปีข้างหน้า
model = LinearRegression() #Choose your Algorithm
year_set2 = about_t.date_range('2018',periods=10,freq='A')
year2_list = year_set2.tolist()

Dframe2_predict = about_t.DataFrame({'Date':year_set2.values})
Dframe2_predict['Date_ordinal'] = Dframe2_predict['Date'].apply(lambda x:x.toordinal())
future_Date = Dframe2_predict[['Date_ordinal']]
Dframe2_val = about_t.DataFrame({'Date':year_set2.values,'Value':model.predict(future_Date)})
model.fit(future_Date,model.predict(future_Date))
mt.plot(year_list+year_set2.tolist(),TH_profit+Dframe2_val['Value'].tolist())
mt.show()
