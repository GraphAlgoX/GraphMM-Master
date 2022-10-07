import numpy as np
import torch


class LocationEncode(torch.nn.Module):
    def __init__(self, loc_dim):
        super(LocationEncode, self).__init__()
        self.lat_freq = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, loc_dim))).float().view(1, -1))
        self.lng_freq = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, loc_dim))).float().view(1, -1))
        self.lat_phase = torch.nn.Parameter(
            torch.zeros(loc_dim).float().view(1, -1))
        self.lng_phase = torch.nn.Parameter(
            torch.zeros(loc_dim).float().view(1, -1))

        # self.dense = torch.nn.Linear(time_dim, expand_dim, bias=False)
        # torch.nn.init.xavier_normal_(self.dense.weight)

    # def forward(self, ts):
    #     # ts: [batch_size, 4]
    #     # 4: min_lat,min_lng,max_lat,max_lng

    #     min_lat = ts[:, [0]]  # [B, 1]
    #     min_lng = ts[:, [1]]  # [B, 1]
    #     max_lat = ts[:, [2]]
    #     max_lng = ts[:, [3]]

    #     # tim = ts[:,:,[2]]
    #     min_lat_ts = torch.cos(min_lat*self.lat_freq + self.phase)
    #     min_lng_ts = torch.cos(min_lng*self.lng_freq + self.phase)

    #     max_lat_ts = torch.cos(max_lat*self.lat_freq + self.phase)
    #     max_lng_ts = torch.cos(max_lng*self.lng_freq + self.phase)

    #     ans = torch.cat(
    #         [min_lat_ts, min_lng_ts, max_lat_ts, max_lng_ts], 1)  # [B, 4]

    #     return ans  # self.dense(harmonic)

    def forward(self, ts):
        """
        ts: [batch_size, [min_lat,min_lng,max_lat,max_lng]]
        """
        batch_size = ts.shape[0]
        lats = ts[:, [0, 2]].T.view(2, batch_size, 1)  # (2, batch, 1)
        lngs = ts[:, [1, 3]].T.view(2, batch_size, 1)

        lat_ts = torch.cos(lats*self.lat_freq + self.lat_phase)  # (2, batch, 4)
        lng_ts = torch.cos(lngs*self.lng_freq + self.lng_phase)
        ans = torch.cat((lat_ts[0], lng_ts[0], lat_ts[1], lng_ts[1]), dim=-1)
        return ans
    
    def trace_gps(self, ts):
        """
        ts: [batch_size, lens, 2]
        """
        batch_size, lens, _ = ts.shape

        lats = ts[:,:,[0]]  # (batch, lens, 1)
        lngs = ts[:,:,[1]]

        lat_ts = torch.cos(lats@self.lat_freq + self.lat_phase)  # (2, batch, 4)
        lng_ts = torch.cos(lngs@self.lng_freq + self.lng_phase)
        ans = torch.cat([lat_ts, lng_ts], dim=-1) # (batch, lens, 2*emb)
        return ans


if __name__ == "__main__":
    pos_emb = LocationEncode(loc_dim=4)
    ts = torch.rand(100, 4)
    emb1 = pos_emb.forward(ts)
    emb2 = pos_emb.forwardMerge(ts)
    print(torch.all(emb1 == emb2))
