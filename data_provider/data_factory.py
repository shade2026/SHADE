from data_provider.data_loader import Dataset_Anomalous_Humidity, Dataset_Anomalous_TrafficFlow, Dataset_Anomalous_NO2, Dataset_Anomalous_Temperature
from torch.utils.data import DataLoader

data_dict = {
    'Humidity_Anomalous': Dataset_Anomalous_Humidity,
    'TrafficFlow_Anomalous': Dataset_Anomalous_TrafficFlow,
    'NO2_Anomalous': Dataset_Anomalous_NO2,
    "Temperature_Anomalous": Dataset_Anomalous_Temperature,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            step=1,  # 新加的, 之前baseline默认100
            # step=args.seq_len,
            flag=flag
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    else:
        if 'Anomalous' in args.data:
            data_set = Data(
                args=args,
                root_path=args.root_path,
                win_size=args.seq_len,
                step=1,  # 新加的, 之前baseline默认100
                # step=args.seq_len,
                flag=flag
            )
        else:
            data_set = Data(
                args=args,
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns
            )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
