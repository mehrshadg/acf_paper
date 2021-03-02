import cifti
from neuro_helper.abstract.map import Space, TopoMap, TemplateMap, TopoName
from neuro_helper.map import ColeTemplateMap, SchaeferTemplateMap, AntPostTopo, MarguliesGradientTopo
from neuro_helper.abstract.map import HierarchyName

__all__ = ["RAW_DATA_ROOT_DIR", "tpt_cole", "tpt_sh", "topo_at", "topo_marg",
           "template_meta_combination"]

from pandas import DataFrame, Series


class CustomAntPostTopo(TopoMap):
    def __init__(self, template: TemplateMap):
        super().__init__(TopoName.ANT_POST_GRADIENT, template)

    def load(self):
        if self.loaded:
            return
        self.template()
        voxels = cifti.read(self.file_full_path)[0][:, self.medial_wall_mask == 0]
        mask_no_wall = self.template.data.mask[self.medial_wall_mask == 0]
        topo = DataFrame({"region": Series(dtype=str), "network": Series(dtype=str),
                          "coord_x": Series(dtype=float), "coord_y": Series(dtype=float),
                          "coord_z": Series(dtype=float)})
        for i, (reg, net) in enumerate(zip(self.template.data.regions, self.template.data.networks)):
            x, y, z = voxels[:, mask_no_wall == i + 1].mean(axis=1)
            topo.loc[i, :] = reg, net, x, y, z
        self._data = topo


RAW_DATA_ROOT_DIR = "/group/northoff/share/mg/hcp/ready"
tpt_sh = SchaeferTemplateMap(Space.K32_CORTEX, 200, 7)()
tpt_cole = ColeTemplateMap(Space.K32_CORTEX)()
topo_at = {
    tpt_sh.key: CustomAntPostTopo(tpt_sh),
    tpt_cole.key: CustomAntPostTopo(tpt_cole),
}
topo_marg = {
    tpt_sh.key: MarguliesGradientTopo(tpt_sh),
    tpt_cole.key: MarguliesGradientTopo(tpt_cole),
}

template_meta_combination = [
    (tpt_sh, HierarchyName.PERIPHERY_CORE),
    (tpt_cole, HierarchyName.EXTENDED_PERIPHERY_CORE),
    (tpt_cole, HierarchyName.RESTRICTED_PERIPHERY_CORE)
]


