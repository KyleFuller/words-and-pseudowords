import train_and_evaluate as _te
import sys as _sys

def main() -> object:
    model = _te.Model(12)
    _, file_path = _sys.argv
    model.load(file_path)

    # not in 194000.txt
    print(model.predict(['antilinear', 'pseudoword', 'peladophobia', 'pneumatosis', 'grivation', 'apricity', 'gymnophobia']))
    print(model.predict(['xiaolongbao', 'qigong', 'wuxia', 'zhengzhi', 'guangzhou', 'xiongmao', 'jiaozi']))
    print(model.predict(['grundschule', 'arbeitsplatz', 'nachrichten', 'wochenende', 'hauptstadt', 'fernseher', 'spielplatz']))
    print(model.predict(['qjkxvpbzt', 'mwcfglhyr', 'zxbqmpfl', 'pkjvwxrt', 'qzfcpxvk', 'kjhbvcxz', 'mxpqwrtz']))
    
    # in 194000.txt
    print(model.predict(['despite', 'however', 'opposable']))
    print(model.predict(['distributed']))

if __name__ == '__main__':
    main()