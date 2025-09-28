import pickle as pkl
from nuscenes.nuscenes import NuScenes
import argparse

def load_nusc_data_infos(dataset, root, info_prefix):
    fname = f'{root}/{info_prefix}_infos_{dataset}.pkl'
    with open(fname, 'rb') as f:
        infos = pkl.load(f)['infos']
    print(f'Loaded {len(infos)} {dataset} samples from {fname}')
    return infos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--info-prefix", type=str)
    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.data_root, verbose=True)

    for split in ["train", "val"]:
        infos = load_nusc_data_infos(split, args.data_root, args.info_prefix)
        city_infos = []
        for info in infos:
            scene_token = nusc.get('sample', info['token'])['scene_token']
            scene = nusc.get('scene', scene_token)
            cn = nusc.get('log', scene['log_token'])['location']
            city_infos.append(cn)

        out_name = f'{args.data_root}/{split}_city_infos_{args.info_prefix}.pkl'
        with open(out_name, 'wb') as f:
            pkl.dump(city_infos, f)
        print(f"Saved {out_name}")


# import pickle as pkl

# from nuscenes.nuscenes import NuScenes


# def load_nusc_data_infos(dataset, root=None):
#     if root is None:
#         root = '/public/MARS/datasets/nuScenes'
#     # with open(f'{root}/nuscenes_infos_temporal_{dataset}.pkl', 'rb') as f:
#     with open(f'{root}/nuScences_map_trainval_infos_{dataset}.pkl', 'rb') as f:
#         infos = pkl.load(f)['infos']
#     print(f'load {len(infos)} {dataset} data infos for nuscenes dataset...')
#     return infos


# if __name__ == '__main__':
#     dataset = 'train'
#     root = '/public/MARS/datasets/nuScenes'
#     nusc = NuScenes(version='v1.0-trainval', dataroot=root, verbose=True)

#     train_infos = load_nusc_data_infos(dataset, root)
#     train_city_infos = []
#     for info in train_infos:
#         scene_token = nusc.get('sample', info['token'])['scene_token']
#         scene = nusc.get('scene', scene_token)
#         cn = nusc.get('log', scene['log_token'])['location']
#         train_city_infos.append(cn)

#     with open(f'{dataset}_city_infos.pkl', 'wb') as f:
#         pkl.dump(train_city_infos, f)

#     dataset = 'val'
#     val_infos = load_nusc_data_infos(dataset, root)
#     val_city_infos = []
#     for info in val_infos:
#         scene_token = nusc.get('sample', info['token'])['scene_token']
#         scene = nusc.get('scene', scene_token)
#         cn = nusc.get('log', scene['log_token'])['location']
#         val_city_infos.append(cn)

#     with open(f'{dataset}_city_infos.pkl', 'wb') as f:
#         pkl.dump(val_city_infos, f)
