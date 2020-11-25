from neuro_helper.abstract.map import Space
from neuro_helper.map import ColeTemplateMap, SchaeferTemplateMap, AntPostTopo, MarguliesGradientTopo
from neuro_helper.abstract.map import HierarchyName, TopoName
import hcp_acf_window as acw
import hcp_acf_zero as acz

__all__ = ["RAW_DATA_ROOT_DIR", "tpt_cole", "tpt_sh", "topo_at", "topo_marg",
           "template_meta_combination", "lib_details"]

RAW_DATA_ROOT_DIR = "/group/northoff/share/mg/hcp/ready"
tpt_sh = SchaeferTemplateMap(Space.K32, 200, 7)()
tpt_cole = ColeTemplateMap(Space.K32)()
topo_at = {
    tpt_sh: AntPostTopo(tpt_sh),
    tpt_cole: AntPostTopo(tpt_cole),
}
topo_marg = {
    tpt_sh: MarguliesGradientTopo(tpt_sh),
    tpt_cole: MarguliesGradientTopo(tpt_cole),
}

template_meta_combination = [
    (tpt_sh, HierarchyName.PERIPHERY_CORE),
    (tpt_cole, HierarchyName.EXTENDED_PERIPHERY_CORE),
    (tpt_cole, HierarchyName.RESTRICTED_PERIPHERY_CORE)
]

lib_details = [
    (acw, "acw", "ACW-50"),
    (acz, "acz", "ACW-0")
]
