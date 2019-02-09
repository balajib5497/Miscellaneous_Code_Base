import numpy as np
import pandas as pd

path = '../Dataset/'

def load_data():
    tp = pd.io.parsers.read_csv(path + 'application_train.csv', iterator=True, chunksize=1000)
    appln_train = pd.concat(tp, ignore_index=True)

    tp = pd.io.parsers.read_csv(path + 'application_test.csv', iterator=True, chunksize=1000)
    appln_test = pd.concat(tp, ignore_index=True)

    tp = pd.io.parsers.read_csv(path + 'bureau.csv', iterator=True, chunksize=1000)
    bureau = pd.concat(tp, ignore_index=True)

    tp = pd.io.parsers.read_csv(path + 'bureau_balance.csv', iterator=True, chunksize=1000)
    bureau_bal = pd.concat(tp, ignore_index=True)

    tp = pd.io.parsers.read_csv(path + 'credit_card_balance.csv', iterator=True, chunksize=1000)
    cc_bal = pd.concat(tp, ignore_index=True)

    tp = pd.io.parsers.read_csv(path + 'installments_payments.csv', iterator=True, chunksize=1000)
    install_pay = pd.concat(tp, ignore_index=True)

    tp = pd.io.parsers.read_csv(path + 'POS_CASH_balance.csv', iterator=True, chunksize=1000)
    POS_bal = pd.concat(tp, ignore_index=True)

    tp = pd.io.parsers.read_csv(path + 'previous_application.csv', iterator=True, chunksize=1000)
    prev_appln = pd.concat(tp, ignore_index=True)
    
    return appln_train, appln_test, bureau, bureau_bal, cc_bal, install_pay, POS_bal, prev_appln


def average_due_day(values):
    count = 0
    for val in values:
        if val not in ['C', 'X', '0']:
            count += int(val)
        
    return count / len(values)


def add_suffix(df, suffix):
    for i in range(len(df.columns)):
        if df.columns[i] not in ['SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']:
            df.rename(columns={df.columns[i]:df.columns[i] + suffix}, inplace=True)


def clean_name(col, x = '/\-: ,', y = '______'):
    return col.translate(str.maketrans(x,y))


appln_train, appln_test, bureau, bureau_bal, cc_bal, install_pay, POS_bal, prev_appln = load_data()


def status_c(value):
    if value == 'C':
        return 1
    return np.nan


def status_x(value):
    if value == 'X':
        return 1
    return np.nan


def status_dpd(value):
    if value != 'C' and value != 'X':
        return float(value)
    return np.nan


# bureau_bal
bureau_bal_p = bureau_bal.copy()

bureau_bal_p['STATUS_C'] = bureau_bal_p['STATUS'].apply(status_c)
bureau_bal_p['STATUS_X'] = bureau_bal_p['STATUS'].apply(status_x)
bureau_bal_p['STATUS_DPD'] = bureau_bal_p['STATUS'].apply(status_dpd)

bureau_bal_p = bureau_bal_p.groupby(by='SK_ID_BUREAU').agg({
    'STATUS_C':np.nanmean,
    'STATUS_X':np.nanmean,
    'STATUS_DPD':np.nanmean
})
    
bureau_bal_p.reset_index(inplace=True)

add_suffix(bureau_bal_p, '_bure_bal')

# bureau 
bureau_p = pd.merge(bureau, bureau_bal_p, on='SK_ID_BUREAU', how='left')
bureau_p.drop(columns=['CREDIT_CURRENCY', 'CREDIT_TYPE'], axis=1, inplace=True)
bureau_p = pd.get_dummies(bureau_p, columns=['CREDIT_ACTIVE'])

bureau_p.columns = [clean_name(col) for col in bureau_p.columns]

bureau_p = bureau_p.groupby(by='SK_ID_CURR').agg({
    'DAYS_CREDIT':np.nanmean,
    'CREDIT_DAY_OVERDUE':np.nanmean,
    'DAYS_CREDIT_ENDDATE':np.nanmean,
    'DAYS_ENDDATE_FACT':np.nanmean,
    'AMT_CREDIT_MAX_OVERDUE':np.nanmean,
    'CNT_CREDIT_PROLONG':np.nanmean,
    'AMT_CREDIT_SUM':np.nanmean,
    'AMT_CREDIT_SUM_DEBT':np.nanmean,
    'AMT_CREDIT_SUM_LIMIT':np.nanmean,
    'AMT_CREDIT_SUM_OVERDUE':np.nanmean,
    'DAYS_CREDIT_UPDATE':np.nanmean,
    'AMT_ANNUITY':np.nanmean,
    'STATUS_C_bure_bal':np.nanmean,
    'STATUS_X_bure_bal':np.nanmean,
    'STATUS_DPD_bure_bal':np.nanmean,
    'CREDIT_ACTIVE_Active':np.nanmean,
    'CREDIT_ACTIVE_Bad_debt':np.nanmean,
    'CREDIT_ACTIVE_Closed':np.nanmean,
    'CREDIT_ACTIVE_Sold':np.nanmean,
})
bureau_p.reset_index(inplace=True)

add_suffix(bureau_p, '_bure')

bureau_p.to_csv('./bureau_p.csv')
# bureau_p = pd.read_csv('./bureau_p.csv')


# POS_bal
POS_bal_p = pd.get_dummies(POS_bal,columns=['NAME_CONTRACT_STATUS'])
POS_bal_p.columns = [clean_name(col) for col in POS_bal_p.columns]
POS_bal_p = POS_bal_p.groupby(by=['SK_ID_CURR','SK_ID_PREV']).agg({
    'CNT_INSTALMENT':np.nanmean,
    'CNT_INSTALMENT_FUTURE':np.nanmean,
    'SK_DPD':np.nanmean,
    'SK_DPD_DEF':np.nanmean,
    'NAME_CONTRACT_STATUS_Active':np.nanmean,
    'NAME_CONTRACT_STATUS_Amortized_debt':np.nanmean,
    'NAME_CONTRACT_STATUS_Approved':np.nanmean,
    'NAME_CONTRACT_STATUS_Canceled':np.nanmean,
    'NAME_CONTRACT_STATUS_Completed':np.nanmean,
    'NAME_CONTRACT_STATUS_Demand':np.nanmean,
    'NAME_CONTRACT_STATUS_Returned_to_the_store':np.nanmean,
    'NAME_CONTRACT_STATUS_Signed':np.nanmean,
    'NAME_CONTRACT_STATUS_XNA':np.nanmean
})
POS_bal_p.reset_index(inplace=True)


# POS_bal_p.drop(columns=['NAME_CONTRACT_STATUS_XNA', 
#                         'NAME_CONTRACT_STATUS_Returned to the store',
#                         'NAME_CONTRACT_STATUS_Demand',
#                         'NAME_CONTRACT_STATUS_Canceled',
#                         'NAME_CONTRACT_STATUS_Approved',
#                         'NAME_CONTRACT_STATUS_Amortized debt'
#                        ],
#               axis=1, inplace=True)

add_suffix(POS_bal_p, '_pos')

prev_appln_p = prev_appln.copy()
add_suffix(prev_appln_p, '_prev')

prev_appln_p = pd.merge(prev_appln_p, POS_bal_p, how='left', on=['SK_ID_CURR', 'SK_ID_PREV'])


# CC_ball

# cc_bal_p = pd.get_dummies(cc_bal,columns=['NAME_CONTRACT_STATUS'])

# def divide_if_not_zero(frame):
#     if frame[1] > 0.0:
#         return frame[0] / frame[1]
#     else:
#         return frame[1]


# cc_bal_p['AMT_DRAWINGS_ATM_CURRENT_p'] = cc_bal_p[['AMT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT']].apply(divide_if_not_zero, axis = 1)
# cc_bal_p['AMT_DRAWINGS_CURRENT_p'] = cc_bal_p[['AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_CURRENT']].apply(divide_if_not_zero, axis = 1)
# cc_bal_p['AMT_DRAWINGS_OTHER_CURRENT_p'] = cc_bal_p[['AMT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT']].apply(divide_if_not_zero, axis = 1)
# cc_bal_p['AMT_DRAWINGS_POS_CURRENT_p'] = cc_bal_p[['AMT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_POS_CURRENT']].apply(divide_if_not_zero, axis = 1)

# cc_bal_p.drop(columns=['MONTHS_BALANCE',
#                        'AMT_DRAWINGS_ATM_CURRENT',
#                        'CNT_DRAWINGS_ATM_CURRENT',
#                        'AMT_DRAWINGS_CURRENT',
#                        'CNT_DRAWINGS_CURRENT',
#                        'AMT_DRAWINGS_OTHER_CURRENT',
#                        'CNT_DRAWINGS_OTHER_CURRENT',
#                        'AMT_DRAWINGS_POS_CURRENT',
#                        'CNT_DRAWINGS_POS_CURRENT'
#                       ], axis=1, inplace=True)

# cc_bal_p.columns


# cc_bal_p = cc_bal_p.groupby(by=['SK_ID_CURR','SK_ID_PREV']).agg({
#     'AMT_BALANCE':np.nanmean,
#     'AMT_CREDIT_LIMIT_ACTUAL':np.nanmean,
#     'AMT_INST_MIN_REGULARITY':np.nanmean,
#     'AMT_PAYMENT_CURRENT':np.nanmean,
#     'AMT_PAYMENT_TOTAL_CURRENT':np.nanmean,
#     'AMT_RECEIVABLE_PRINCIPAL':np.nanmean,
#     'AMT_RECIVABLE':np.nanmean,
#     'AMT_TOTAL_RECEIVABLE':np.nanmean,
#     'CNT_INSTALMENT_MATURE_CUM':np.nanmean,
#     'SK_DPD':np.nanmean,
#     'SK_DPD_DEF':np.nanmean,
#     'NAME_CONTRACT_STATUS_Active':np.nanmean,
#     'NAME_CONTRACT_STATUS_Approved':np.nanmean,
#     'NAME_CONTRACT_STATUS_Completed':np.nanmean,
#     'NAME_CONTRACT_STATUS_Demand':np.nanmean,
#     'NAME_CONTRACT_STATUS_Refused':np.nanmean,
#     'NAME_CONTRACT_STATUS_Sent proposal':np.nanmean,
#     'NAME_CONTRACT_STATUS_Signed':np.nanmean,
#     'AMT_DRAWINGS_ATM_CURRENT_p':np.nanmean,
#     'AMT_DRAWINGS_CURRENT_p':np.nanmean,
#     'AMT_DRAWINGS_OTHER_CURRENT_p':np.nanmean,
#     'AMT_DRAWINGS_POS_CURRENT_p':np.nanmean
# })
# cc_bal_p.reset_index(inplace=True)



# prev_appln_p = pd.merge(prev_appln_p, cc_bal_p, how='left', on=['SK_ID_CURR', 'SK_ID_PREV'])


# ## install_pay

# In[33]:


install_pay_p = install_pay.copy()


# In[34]:


install_pay_p.columns


# In[35]:


install_pay_p['LATE_PAY_DAYS'] = install_pay_p['DAYS_INSTALMENT'] - install_pay_p['DAYS_ENTRY_PAYMENT']
install_pay_p['REPAY_GAP'] = install_pay_p['AMT_INSTALMENT'] - install_pay_p['AMT_PAYMENT']


# In[36]:


install_pay_p = install_pay_p.groupby(by=['SK_ID_CURR','SK_ID_PREV']).agg({
    'NUM_INSTALMENT_VERSION':np.nanmean,
    'NUM_INSTALMENT_NUMBER':np.nanmean,
    'LATE_PAY_DAYS':np.nanmean,
    'REPAY_GAP':np.nanmean
})
install_pay_p.reset_index(inplace=True)


# In[37]:


add_suffix(install_pay_p, '_inst')


# In[38]:


install_pay_p.columns


# In[39]:


prev_appln_p = pd.merge(prev_appln_p, install_pay_p, how='left', on=['SK_ID_CURR', 'SK_ID_PREV'])


# ## prev_appln

# In[40]:


def yes_rate(values):
    count = 0
    for val in values:
        if val == 'Y':
            count += 1
            
    return count / len(values)


# In[41]:


prev_appln_p['DAYS_DUE_GAP_prev'] = prev_appln_p['DAYS_LAST_DUE_1ST_VERSION_prev'] - prev_appln_p['DAYS_FIRST_DUE_prev']
prev_appln_p['DAYS_TERMINATION_GAP_prev'] = prev_appln_p['DAYS_TERMINATION_prev'] - prev_appln_p['DAYS_LAST_DUE_prev']


# In[42]:


prev_appln_p = pd.get_dummies(prev_appln_p,columns=['NAME_CONTRACT_TYPE_prev',
                                                    'WEEKDAY_APPR_PROCESS_START_prev',
                                                    'NAME_CASH_LOAN_PURPOSE_prev',
                                                    'NAME_CONTRACT_STATUS_prev',
                                                    'NAME_PAYMENT_TYPE_prev',
                                                    'CODE_REJECT_REASON_prev',
                                                    'NAME_TYPE_SUITE_prev',
                                                    'NAME_CLIENT_TYPE_prev',
                                                    'NAME_GOODS_CATEGORY_prev',
                                                    'NAME_PORTFOLIO_prev',
                                                    'NAME_PRODUCT_TYPE_prev',
                                                    'CHANNEL_TYPE_prev',
                                                    'NAME_SELLER_INDUSTRY_prev',
                                                    'NAME_YIELD_GROUP_prev',
                                                    'PRODUCT_COMBINATION_prev'
                                                   ])


# In[43]:


prev_appln_p.columns = [clean_name(col) for col in prev_appln_p.columns]


# In[44]:


prev_appln_p = prev_appln_p.groupby(by='SK_ID_CURR').agg({
    'AMT_ANNUITY_prev':np.nanmean,
    'AMT_APPLICATION_prev':np.nanmean,
    'AMT_CREDIT_prev':np.nanmean,
    'AMT_DOWN_PAYMENT_prev':np.nanmean,
    'AMT_GOODS_PRICE_prev':np.nanmean,
    'FLAG_LAST_APPL_PER_CONTRACT_prev':yes_rate,
    'HOUR_APPR_PROCESS_START_prev':np.nanmean,
    'NFLAG_LAST_APPL_IN_DAY_prev':np.nanmean,
    'RATE_DOWN_PAYMENT_prev':np.nanmean,
    'DAYS_DECISION_prev':np.nanmean,
    'CNT_PAYMENT_prev':np.nanmean,
    'DAYS_DUE_GAP_prev':np.nanmean,
    'DAYS_TERMINATION_GAP_prev':np.nanmean,
    'NFLAG_INSURED_ON_APPROVAL_prev':np.nanmean,
    
    'CNT_INSTALMENT_pos':np.nanmean,
    'CNT_INSTALMENT_FUTURE_pos':np.nanmean,
    'SK_DPD_pos':np.nanmean,
    'SK_DPD_DEF_pos':np.nanmean,
    'NAME_CONTRACT_STATUS_Active_pos':np.nanmean,
    'NAME_CONTRACT_STATUS_Amortized_debt_pos':np.nanmean,
    'NAME_CONTRACT_STATUS_Approved_pos':np.nanmean,
    'NAME_CONTRACT_STATUS_Canceled_pos':np.nanmean,
    'NAME_CONTRACT_STATUS_Completed_pos':np.nanmean,
    'NAME_CONTRACT_STATUS_Demand_pos':np.nanmean,
    'NAME_CONTRACT_STATUS_Returned_to_the_store_pos':np.nanmean,
    'NAME_CONTRACT_STATUS_Signed_pos':np.nanmean,
    'NAME_CONTRACT_STATUS_XNA_pos':np.nanmean,
    
    'NUM_INSTALMENT_VERSION_inst':np.nanmean,
    'NUM_INSTALMENT_NUMBER_inst':np.nanmean,
    'LATE_PAY_DAYS_inst':np.nanmean,
    'REPAY_GAP_inst':np.nanmean,
    
    'NAME_CONTRACT_TYPE_prev_Cash_loans':np.nanmean,
    'NAME_CONTRACT_TYPE_prev_Consumer_loans':np.nanmean,
    'NAME_CONTRACT_TYPE_prev_Revolving_loans':np.nanmean,
    'NAME_CONTRACT_TYPE_prev_XNA':np.nanmean,
    
    'WEEKDAY_APPR_PROCESS_START_prev_FRIDAY':np.nanmean,
    'WEEKDAY_APPR_PROCESS_START_prev_MONDAY':np.nanmean,
    'WEEKDAY_APPR_PROCESS_START_prev_SATURDAY':np.nanmean,
    'WEEKDAY_APPR_PROCESS_START_prev_SUNDAY':np.nanmean,
    'WEEKDAY_APPR_PROCESS_START_prev_THURSDAY':np.nanmean,
    'WEEKDAY_APPR_PROCESS_START_prev_TUESDAY':np.nanmean,
    'WEEKDAY_APPR_PROCESS_START_prev_WEDNESDAY':np.nanmean,
    
    'NAME_CASH_LOAN_PURPOSE_prev_Building_a_house_or_an_annex':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Business_development':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Buying_a_garage':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Buying_a_holiday_home___land':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Buying_a_home':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Buying_a_new_car':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Buying_a_used_car':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Car_repairs':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Education':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Everyday_expenses':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Furniture':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Gasification___water_supply':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Hobby':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Journey':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Medicine':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Money_for_a_third_person':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Other':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Payments_on_other_loans':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Purchase_of_electronic_equipment':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Refusal_to_name_the_goal':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Repairs':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Urgent_needs':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_Wedding___gift___holiday':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_XAP':np.nanmean,
    'NAME_CASH_LOAN_PURPOSE_prev_XNA':np.nanmean,
    
    'NAME_CONTRACT_STATUS_prev_Approved':np.nanmean,
    'NAME_CONTRACT_STATUS_prev_Canceled':np.nanmean,
    'NAME_CONTRACT_STATUS_prev_Refused':np.nanmean,
    'NAME_CONTRACT_STATUS_prev_Unused_offer':np.nanmean,
    
    'NAME_PAYMENT_TYPE_prev_Cash_through_the_bank':np.nanmean,
    'NAME_PAYMENT_TYPE_prev_Cashless_from_the_account_of_the_employer':np.nanmean,
    'NAME_PAYMENT_TYPE_prev_Non_cash_from_your_account':np.nanmean,
    'NAME_PAYMENT_TYPE_prev_XNA':np.nanmean,

    'CODE_REJECT_REASON_prev_CLIENT':np.nanmean,
    'CODE_REJECT_REASON_prev_HC':np.nanmean,
    'CODE_REJECT_REASON_prev_LIMIT':np.nanmean,
    'CODE_REJECT_REASON_prev_SCO':np.nanmean,
    'CODE_REJECT_REASON_prev_SCOFR':np.nanmean,
    'CODE_REJECT_REASON_prev_SYSTEM':np.nanmean,
    'CODE_REJECT_REASON_prev_VERIF':np.nanmean,
    'CODE_REJECT_REASON_prev_XAP':np.nanmean,
    'CODE_REJECT_REASON_prev_XNA':np.nanmean,
    
    'NAME_TYPE_SUITE_prev_Children':np.nanmean,
    'NAME_TYPE_SUITE_prev_Family':np.nanmean,
    'NAME_TYPE_SUITE_prev_Group_of_people':np.nanmean,
    'NAME_TYPE_SUITE_prev_Other_A':np.nanmean,
    'NAME_TYPE_SUITE_prev_Other_B':np.nanmean,
    'NAME_TYPE_SUITE_prev_Spouse__partner':np.nanmean,
    'NAME_TYPE_SUITE_prev_Unaccompanied':np.nanmean,
    
    'NAME_CLIENT_TYPE_prev_New':np.nanmean,
    'NAME_CLIENT_TYPE_prev_Refreshed':np.nanmean,
    'NAME_CLIENT_TYPE_prev_Repeater':np.nanmean,
    'NAME_CLIENT_TYPE_prev_XNA':np.nanmean,
    
    'NAME_GOODS_CATEGORY_prev_Additional_Service':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Animals':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Audio_Video':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Auto_Accessories':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Clothing_and_Accessories':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Computers':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Construction_Materials':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Consumer_Electronics':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Direct_Sales':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Education':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Fitness':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Furniture':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Gardening':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Homewares':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_House_Construction':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Insurance':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Jewelry':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Medical_Supplies':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Medicine':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Mobile':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Office_Appliances':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Other':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Photo___Cinema_Equipment':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Sport_and_Leisure':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Tourism':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Vehicles':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_Weapon':np.nanmean,
    'NAME_GOODS_CATEGORY_prev_XNA':np.nanmean,
    
    'NAME_PORTFOLIO_prev_Cards':np.nanmean,
    'NAME_PORTFOLIO_prev_Cars':np.nanmean,
    'NAME_PORTFOLIO_prev_Cash':np.nanmean,
    'NAME_PORTFOLIO_prev_POS':np.nanmean,
    'NAME_PORTFOLIO_prev_XNA':np.nanmean,
    
    'NAME_PRODUCT_TYPE_prev_XNA':np.nanmean,
    'NAME_PRODUCT_TYPE_prev_walk_in':np.nanmean,
    'NAME_PRODUCT_TYPE_prev_x_sell':np.nanmean,
    
    'CHANNEL_TYPE_prev_AP+_(Cash_loan)':np.nanmean,
    'CHANNEL_TYPE_prev_Car_dealer':np.nanmean,
    'CHANNEL_TYPE_prev_Channel_of_corporate_sales':np.nanmean,
    'CHANNEL_TYPE_prev_Contact_center':np.nanmean,
    'CHANNEL_TYPE_prev_Country_wide':np.nanmean,
    'CHANNEL_TYPE_prev_Credit_and_cash_offices':np.nanmean,
    'CHANNEL_TYPE_prev_Regional___Local':np.nanmean,
    'CHANNEL_TYPE_prev_Stone':np.nanmean,
    
    'NAME_SELLER_INDUSTRY_prev_Auto_technology':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_Clothing':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_Connectivity':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_Construction':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_Consumer_electronics':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_Furniture':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_Industry':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_Jewelry':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_MLM_partners':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_Tourism':np.nanmean,
    'NAME_SELLER_INDUSTRY_prev_XNA':np.nanmean,
    
    'NAME_YIELD_GROUP_prev_XNA':np.nanmean,
    'NAME_YIELD_GROUP_prev_high':np.nanmean,
    'NAME_YIELD_GROUP_prev_low_action':np.nanmean,
    'NAME_YIELD_GROUP_prev_low_normal':np.nanmean,
    'NAME_YIELD_GROUP_prev_middle':np.nanmean,
    
    'PRODUCT_COMBINATION_prev_Card_Street':np.nanmean,
    'PRODUCT_COMBINATION_prev_Card_X_Sell':np.nanmean,
    'PRODUCT_COMBINATION_prev_Cash':np.nanmean,
    'PRODUCT_COMBINATION_prev_Cash_Street__high':np.nanmean,
    'PRODUCT_COMBINATION_prev_Cash_Street__low':np.nanmean,
    'PRODUCT_COMBINATION_prev_Cash_Street__middle':np.nanmean,
    'PRODUCT_COMBINATION_prev_Cash_X_Sell__high':np.nanmean,
    'PRODUCT_COMBINATION_prev_Cash_X_Sell__low':np.nanmean,
    'PRODUCT_COMBINATION_prev_Cash_X_Sell__middle':np.nanmean,
    'PRODUCT_COMBINATION_prev_POS_household_with_interest':np.nanmean,
    'PRODUCT_COMBINATION_prev_POS_household_without_interest':np.nanmean,
    'PRODUCT_COMBINATION_prev_POS_industry_with_interest':np.nanmean,
    'PRODUCT_COMBINATION_prev_POS_industry_without_interest':np.nanmean,
    'PRODUCT_COMBINATION_prev_POS_mobile_with_interest':np.nanmean,
    'PRODUCT_COMBINATION_prev_POS_mobile_without_interest':np.nanmean,
    'PRODUCT_COMBINATION_prev_POS_other_with_interest':np.nanmean,
    'PRODUCT_COMBINATION_prev_POS_others_without_interest':np.nanmean
})
prev_appln_p.reset_index(inplace=True)


# In[45]:


prev_appln_p.to_csv('./prev_appln_p.csv')


# In[46]:


appln_tmp =  pd.merge(appln_train, bureau_p, how='left', on='SK_ID_CURR')
appln_tmp =  pd.merge(appln_tmp, prev_appln_p, how='left', on='SK_ID_CURR')


# In[47]:


appln_tmp.to_csv('./appln_tmp.csv')


# ## appln_train

# In[57]:


appln_tmp[(appln_tmp['ENTRANCES_AVG'].isnull() ^ False) & 
          (appln_tmp['ENTRANCES_MEDI'].isnull() ^ True) &
          (appln_tmp['ENTRANCES_MODE'].isnull() ^ True)
         ].shape


# In[61]:


count = appln_tmp.isnull().sum()
pers = count * 100/ appln_tmp.shape[0]
for col, per in zip(appln_tmp.columns, pers):
    print(col, '\t\t\t', per)


# In[59]:


for col in appln_tmp.columns:
    print(col)


# In[60]:


appln_tmp.STATUS_C_bure_bal_bure


# In[63]:


count = bureau_p.isnull().sum()
count * 100/ bureau_p.shape[0]

