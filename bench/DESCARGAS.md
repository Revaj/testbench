# Descarga manual de los 38 datasets de clasificación de TabArena-v0.1

Lista generada desde `bench/curated_tabarena_dataset_metadata.csv` (copia sin modificar de
[autogluon/tabarena](https://github.com/autogluon/tabarena), Apache-2.0). Misma tabla en `bench/descargas.csv`.

- **Página del dataset** (`openml.org/d/<id>`): botón de descarga (ARFF, CSV o Parquet).
- **Particiones**: archivo ARFF con train/test oficiales de cada task. OpenML trae 10 repeticiones;
  TabArena-v0.1 usa solo las primeras `rep` (3 en datasets de ≥ 2,500 filas). `fetch_data.py` ya filtra eso.
- Lo más simple sigue siendo `python bench/fetch_data.py --jobs=8`: baja y verifica todo, en paralelo, y retoma si se corta.

| # | Dataset | Tipo | Filas | Features | Clases | rep | Dataset | Particiones |
|---|---|---|---|---|---|---|---|---|
| 1 | blood-transfusion-service-center | binary | 748 | 4 | 2 | 10 | [d/46913](https://www.openml.org/d/46913) | [task 363621](https://www.openml.org/api_splits/get/363621/Task_363621_splits.arff) |
| 2 | diabetes | binary | 768 | 8 | 2 | 10 | [d/46921](https://www.openml.org/d/46921) | [task 363629](https://www.openml.org/api_splits/get/363629/Task_363629_splits.arff) |
| 3 | anneal | multiclass | 898 | 38 | 5 | 10 | [d/46906](https://www.openml.org/d/46906) | [task 363614](https://www.openml.org/api_splits/get/363614/Task_363614_splits.arff) |
| 4 | credit-g | binary | 1,000 | 20 | 2 | 10 | [d/46918](https://www.openml.org/d/46918) | [task 363626](https://www.openml.org/api_splits/get/363626/Task_363626_splits.arff) |
| 5 | maternal_health_risk | multiclass | 1,014 | 6 | 3 | 10 | [d/46941](https://www.openml.org/d/46941) | [task 363685](https://www.openml.org/api_splits/get/363685/Task_363685_splits.arff) |
| 6 | qsar-biodeg | binary | 1,054 | 41 | 2 | 10 | [d/46952](https://www.openml.org/d/46952) | [task 363696](https://www.openml.org/api_splits/get/363696/Task_363696_splits.arff) |
| 7 | website_phishing | multiclass | 1,353 | 9 | 3 | 10 | [d/46963](https://www.openml.org/d/46963) | [task 363707](https://www.openml.org/api_splits/get/363707/Task_363707_splits.arff) |
| 8 | Fitness_Club | binary | 1,500 | 6 | 2 | 10 | [d/46927](https://www.openml.org/d/46927) | [task 363671](https://www.openml.org/api_splits/get/363671/Task_363671_splits.arff) |
| 9 | MIC | multiclass | 1,699 | 111 | 8 | 10 | [d/46980](https://www.openml.org/d/46980) | [task 363711](https://www.openml.org/api_splits/get/363711/Task_363711_splits.arff) |
| 10 | Is-this-a-good-customer | binary | 1,723 | 13 | 2 | 10 | [d/46938](https://www.openml.org/d/46938) | [task 363682](https://www.openml.org/api_splits/get/363682/Task_363682_splits.arff) |
| 11 | Marketing_Campaign | binary | 2,240 | 25 | 2 | 10 | [d/46940](https://www.openml.org/d/46940) | [task 363684](https://www.openml.org/api_splits/get/363684/Task_363684_splits.arff) |
| 12 | hazelnut-spread-contaminant-detection | binary | 2,400 | 30 | 2 | 10 | [d/46930](https://www.openml.org/d/46930) | [task 363674](https://www.openml.org/api_splits/get/363674/Task_363674_splits.arff) |
| 13 | seismic-bumps | binary | 2,584 | 15 | 2 | 3 | [d/46956](https://www.openml.org/d/46956) | [task 363700](https://www.openml.org/api_splits/get/363700/Task_363700_splits.arff) |
| 14 | splice | multiclass | 3,190 | 60 | 3 | 3 | [d/46958](https://www.openml.org/d/46958) | [task 363702](https://www.openml.org/api_splits/get/363702/Task_363702_splits.arff) |
| 15 | Bioresponse | binary | 3,751 | 1776 | 2 | 3 | [d/46912](https://www.openml.org/d/46912) | [task 363620](https://www.openml.org/api_splits/get/363620/Task_363620_splits.arff) |
| 16 | hiva_agnostic | multiclass | 3,845 | 1617 | 3 | 3 | [d/46933](https://www.openml.org/d/46933) | [task 363677](https://www.openml.org/api_splits/get/363677/Task_363677_splits.arff) |
| 17 | students_dropout_and_academic_success | multiclass | 4,424 | 36 | 3 | 3 | [d/46960](https://www.openml.org/d/46960) | [task 363704](https://www.openml.org/api_splits/get/363704/Task_363704_splits.arff) |
| 18 | churn | binary | 5,000 | 19 | 2 | 3 | [d/46915](https://www.openml.org/d/46915) | [task 363623](https://www.openml.org/api_splits/get/363623/Task_363623_splits.arff) |
| 19 | polish_companies_bankruptcy | binary | 5,910 | 64 | 2 | 3 | [d/46950](https://www.openml.org/d/46950) | [task 363694](https://www.openml.org/api_splits/get/363694/Task_363694_splits.arff) |
| 20 | taiwanese_bankruptcy_prediction | binary | 6,819 | 94 | 2 | 3 | [d/46962](https://www.openml.org/d/46962) | [task 363706](https://www.openml.org/api_splits/get/363706/Task_363706_splits.arff) |
| 21 | NATICUSdroid | binary | 7,491 | 86 | 2 | 3 | [d/46969](https://www.openml.org/d/46969) | [task 363689](https://www.openml.org/api_splits/get/363689/Task_363689_splits.arff) |
| 22 | coil2000_insurance_policies | binary | 9,822 | 85 | 2 | 3 | [d/46916](https://www.openml.org/d/46916) | [task 363624](https://www.openml.org/api_splits/get/363624/Task_363624_splits.arff) |
| 23 | Bank_Customer_Churn | binary | 10,000 | 10 | 2 | 3 | [d/46911](https://www.openml.org/d/46911) | [task 363619](https://www.openml.org/api_splits/get/363619/Task_363619_splits.arff) |
| 24 | heloc | binary | 10,459 | 23 | 2 | 3 | [d/46932](https://www.openml.org/d/46932) | [task 363676](https://www.openml.org/api_splits/get/363676/Task_363676_splits.arff) |
| 25 | jm1 | binary | 10,885 | 21 | 2 | 3 | [d/46979](https://www.openml.org/d/46979) | [task 363712](https://www.openml.org/api_splits/get/363712/Task_363712_splits.arff) |
| 26 | E-CommereShippingData | binary | 10,999 | 10 | 2 | 3 | [d/46924](https://www.openml.org/d/46924) | [task 363632](https://www.openml.org/api_splits/get/363632/Task_363632_splits.arff) |
| 27 | online_shoppers_intention | binary | 12,330 | 17 | 2 | 3 | [d/46947](https://www.openml.org/d/46947) | [task 363691](https://www.openml.org/api_splits/get/363691/Task_363691_splits.arff) |
| 28 | in_vehicle_coupon_recommendation | binary | 12,684 | 24 | 2 | 3 | [d/46937](https://www.openml.org/d/46937) | [task 363681](https://www.openml.org/api_splits/get/363681/Task_363681_splits.arff) |
| 29 | HR_Analytics_Job_Change_of_Data_Scientists | binary | 19,158 | 12 | 2 | 3 | [d/46935](https://www.openml.org/d/46935) | [task 363679](https://www.openml.org/api_splits/get/363679/Task_363679_splits.arff) |
| 30 | credit_card_clients_default | binary | 30,000 | 23 | 2 | 3 | [d/46919](https://www.openml.org/d/46919) | [task 363627](https://www.openml.org/api_splits/get/363627/Task_363627_splits.arff) |
| 31 | Amazon_employee_access | binary | 32,769 | 9 | 2 | 3 | [d/46905](https://www.openml.org/d/46905) | [task 363613](https://www.openml.org/api_splits/get/363613/Task_363613_splits.arff) |
| 32 | bank-marketing | binary | 45,211 | 13 | 2 | 3 | [d/46910](https://www.openml.org/d/46910) | [task 363618](https://www.openml.org/api_splits/get/363618/Task_363618_splits.arff) |
| 33 | kddcup09_appetency | binary | 50,000 | 212 | 2 | 3 | [d/46939](https://www.openml.org/d/46939) | [task 363683](https://www.openml.org/api_splits/get/363683/Task_363683_splits.arff) |
| 34 | Diabetes130US | binary | 71,518 | 47 | 2 | 3 | [d/46922](https://www.openml.org/d/46922) | [task 363630](https://www.openml.org/api_splits/get/363630/Task_363630_splits.arff) |
| 35 | APSFailure | binary | 76,000 | 170 | 2 | 3 | [d/46908](https://www.openml.org/d/46908) | [task 363616](https://www.openml.org/api_splits/get/363616/Task_363616_splits.arff) |
| 36 | SDSS17 | multiclass | 78,053 | 11 | 3 | 3 | [d/46955](https://www.openml.org/d/46955) | [task 363699](https://www.openml.org/api_splits/get/363699/Task_363699_splits.arff) |
| 37 | customer_satisfaction_in_airline | binary | 129,880 | 21 | 2 | 3 | [d/46920](https://www.openml.org/d/46920) | [task 363628](https://www.openml.org/api_splits/get/363628/Task_363628_splits.arff) |
| 38 | GiveMeSomeCredit | binary | 150,000 | 10 | 2 | 3 | [d/46929](https://www.openml.org/d/46929) | [task 363673](https://www.openml.org/api_splits/get/363673/Task_363673_splits.arff) |
