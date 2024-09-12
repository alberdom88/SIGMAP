
KNIME=knime
MODEL=./SIGMApred_final_final.knwf.knwf
MODEL1=./parte1_final_final.knwf.knwf

linee=$(wc -l < $1)

# Controlla se il numero di linee è maggiore di 1
if [ "$linee" -gt 1 ]; then
   $KNIME --launcher.suppressErrors -nosplash -nosave -reset -application org.knime.product.KNIME_BATCH_APPLICATION -workflowFile="$MODEL" -workflow.variable=path,$(pwd)/$1,String -workflow.variable=path_1,$(pwd)/$1_out1.csv,String
else
   $KNIME --launcher.suppressErrors -nosplash -nosave -reset -application org.knime.product.KNIME_BATCH_APPLICATION -workflowFile="$MODEL1" -workflow.variable=path,$(pwd)/$1,String -workflow.variable=path_1,$(pwd)/$1_out,String
   python3 shap_morgan.py $1_out
   awk '{print $2}' $1 > $1_to_dela.smi
   python3 ./DelaDrugS/generate_v3.py $1_to_dela.smi 10 1
   awk '{print NR", "$1}' $1_to_dela.smi_output > $1_counter.smi
   $KNIME --launcher.suppressErrors -nosplash -nosave -reset -application org.knime.product.KNIME_BATCH_APPLICATION -workflowFile="$MODEL" -workflow.variable=path,$(pwd)/$1_counter.smi,String -workflow.variable=path_1,$(pwd)/$1_counter_out.csv,String
fi

