from enum import Enum

class Violation(Enum):
    # Accelerating
    VIO_SPD_LIM = 0
    VIO_BSC_SPD = 1
    
    # Direction Changing
    UNLAW_LANE_CHG = 2
    F_MERGING = 3
    DANGER_LFT_TURN = 4
    F_SPEC_LFT_TURN = 5
    ILL_U_TURN = 6
    IMPR_LFT_TURN = 7
    F_UNCTRL_T_INTER = 8
    IMPR_RGT_TURN = 9
    UNLAW_TURN = 10
    
    # Backing
    ILL_BACK = 11
    
    # Passing
    UNSF_PASS_LFT = 12
    UNSF_PASS_RGT = 13
    F_RNDABT = 14
    PASS_NP_ZONE = 15
    PASS_ST_VEH_CRWK = 16
    PASS_BIC = 17
    
    # Cruising
    CR_CENT_LN = 18
    DRV_ON_DIVIDER = 19
    DRV_LFT_SPEC_ZONE = 20
    DRV_SF_ZONE = 21
    F_DRV_RGT_DHW = 22
    F_OBEY_ONE_WAY = 23
    OP_BIC_TRAIL = 24
    UNLAW_OP_LOW_SPD_HW = 25
    FL_TOO_CLOSE = 26
    F_DRV_RGT_APPR_VEH = 27
    OBST_CR_TRF = 28
    F_DRV_ENTER = 29
    DEPR_FULL_LANE = 30
    F_DRV_IN_LANE = 31
    F_SLOW_DRV_RGT = 32
    F_SLOW_DRV_OVTK_VEH = 33
    
    # Overlap
    GREEN_LGT = 34
    GREEN_ARW_LGT = 35
    STD_YEL_CIR_LGT = 36
    STD_YEL_ARW_LGT = 37
    FLS_YEL_CIR_LGT = 38
    FLS_YEL_ARW_LGT = 39
    STD_RED_CIR_LGT = 40
    STD_RED_ARW_LGT = 41
    FLS_RED_CIR_LGT = 42
    STOP_SIG = 43
    F_USE_SIG_RNDABT = 44
    F_SIG_LGT = 45
    F_UNCTRL_INTER = 46
    F_ST_MERGING = 47
    F_BIC_SW = 48
    F_PED_SW = 49
    DRV_WRONG_TRF_ISD = 50
    IMPR_USE_CENT_LANE = 51
    F_YIELD_BIC_LANE = 52
    F_ST_PED = 53
    F_DRV_RGT = 54
    
    # Deceleration
    IMD_TRF = 55
    UNLAW_ST_DEC = 56
    
    # Other
    CARELESS = 57
    RECKLESS = 58
    MISUSE_SPEC_LFT_TURN = 59