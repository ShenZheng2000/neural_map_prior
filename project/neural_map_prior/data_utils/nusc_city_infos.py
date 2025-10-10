import pickle as pkl
import os
from nuscenes.nuscenes import NuScenes


def load_nusc_data_infos(dataset, root=None):
    if root is None:
        root = '/public/MARS/datasets/nuScenes'
    with open(f'{root}/nuScences_map_trainval_infos_{dataset}.pkl', 'rb') as f:
        infos = pkl.load(f)['infos']
    print(f'load {len(infos)} {dataset} data infos for nuscenes dataset...')
    return infos


if __name__ == '__main__':
    root = "data/nuscenes"
    nusc = NuScenes(version='v1.0-trainval', dataroot=root, verbose=True)

    for dataset in ['train', 'val']:
        infos = load_nusc_data_infos(dataset, root)
        city_infos = []
        for info in infos:
            scene_token = nusc.get('sample', info['token'])['scene_token']
            scene = nusc.get('scene', scene_token)
            cn = nusc.get('log', scene['log_token'])['location']
            city_infos.append(cn)

        out_path = os.path.join(root, f"{dataset}_city_infos.pkl")
        with open(out_path, 'wb') as f:
            pkl.dump(city_infos, f)
        print(f"✅ Saved {out_path}")