method=''


while getopts 'm:' flag; do
    case "${flag}" in
        m) method="${OPTARG}" ;;
    esac
done

case "$method" in
    "bert") python bert.py;
            python test.py;;
    "bert_DNN") python bert.py;
                python words_embedding.py false;
                python words_embedding.py true;
                python sentence_embedding.py false;
                python sentence_embedding.py true;
                python bert_DNN.py;;
    "bert_KNN") python bert.py;
                python words_embedding.py false;
                python words_embedding.py true;
                python sentence_embedding.py false;
                python sentence_embedding.py true;
                python bert_KNN.py;;
    "tf-idf") python TF_IDF.py;;
    "bigram") python bigram.py;;
esac