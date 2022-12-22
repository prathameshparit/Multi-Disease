import pickle
import decimal
import numpy as np

insurance_model = pickle.load(open('models/Insurance.sav', 'rb'))


def currencyInIndiaFormat(n):
    d = decimal.Decimal(str(n))
    if d.as_tuple().exponent < -2:
        s = str(n)
    else:
        s = '{0:.2f}'.format(n)
    l = len(s)
    i = l - 1;
    res = ''
    flag = 0
    k = 0
    while i >= 0:
        if flag == 0:
            res = res + s[i]
            if s[i] == '.':
                flag = 1
        elif flag == 1:
            k = k + 1
            res = res + s[i]
            if k == 3 and i - 1 >= 0:
                res = res + ','
                flag = 2
                k = 0
        else:
            k = k + 1
            res = res + s[i]
            if k == 2 and i - 1 >= 0:
                res = res + ','
                flag = 2
                k = 0
        i = i - 1

    return res[::-1]


def insurance_predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = insurance_model.predict(input_data_reshaped)
    # print(prediction)

    # print('The insurance cost is Indian Rupees ₹', str(round(prediction[0], 2)))

    return currencyInIndiaFormat(round(prediction[0], 2))












#
# print(insurance_predict(input_data))

# (['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'insurance'], dtype='object')


# {'sex':{'male':0,'female':1}}
#
# 3 # encoding 'smoker' column
# {'smoker':{'yes':0,'no':1}}
#
# # encoding 'region' column
# {'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}
