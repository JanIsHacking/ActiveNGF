import torch

from jans_scripts.visualize import setup_rerun, visualize_pc

def main():
    setup_rerun("graspnet_batch_viz")

    batch_data = torch.load("jans_scripts/data/batch_data.pt")
    coors = batch_data["coors"]
    feats = batch_data["feats"]
    pc = batch_data["point_clouds"][0]
    colors = batch_data["pcd_color"][0][:, :3]

    print(coors.shape)
    print(feats.shape)
    print(pc.shape)
    print(colors.shape)

    print(coors[:5, :])
    print(feats[:5, :])
    print(pc[:5, :])
    print(colors[:5, :])

    #normal_colors = feats[:, 1:].float().mean(dim=1).repeat(3, 1).T
    
    visualize_pc(pc, colors, suffix="base")

if __name__ == "__main__":
    main()