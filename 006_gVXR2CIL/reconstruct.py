has_cil = True
try:
    from cil.processors import TransmissionAbsorptionConverter
    
    # FBP with Tigre or Astra
    from cil.recon import FBP as FBP_recons_cil

    # FDK with Tigre only
    from cil.recon import FDK as FDK_recons_cil # Tigre only

    # Do not use
    # from cil.plugins.astra.processors.FDK_Flexible import FDK_Flexible

    # FBP and FDK with Astra
    from cil.plugins.astra import FBP as FBP_plugin_astra

    print("CIL detected")
except:
    has_cil = False
    print("CIL not detected")


has_tigre = True
try:
    # FBP and FDK with Tigre
    from cil.plugins.tigre import FBP as FBP_plugin_tigre
    print("Tigre detected")
except:
    has_tigre = False
    print("Tigre not detected")

    
    
    
def reconstructFBPWithCIL(data, ig, use_plugins, verbose):
    if verbose > 0: print("Parallel beam detected")


    if has_tigre:
        if verbose > 0: print("Backend: Tigre")
        if verbose > 0: print("Use plugin directly: ", use_plugins)

        if use_plugins:
            reconstruction:ImageData | None = FBP_plugin_tigre(ig,data.geometry)(data)
        else:
            reconstruction:ImageData | None = FBP_recons_cil(data, ig, backend="tigre").run()

    else:
        if verbose > 0: print("Backend: Astra-Toolbox")
        if verbose > 0: print("Use plugin directly: ", use_plugins)

        if use_plugins:
            reconstruction:ImageData | None = FBP_plugin_astra(ig,data.geometry)(data)
        else:
            reconstruction:ImageData | None = FBP_recons_cil(data, ig, backend="astra").run()

    return reconstruction


def reconstructFDKWithCIL(data, ig, use_plugins, verbose):
    if verbose > 0: print("Cone beam detected")

    # if has_tigre:
    #     if verbose > 0: print("Backend: Tigre")
    #     reconstruction:ImageData | None = FDK(data, ig).run()
    # else:
    #     if verbose > 0: print("Backend: Astra-Toolbox")
    #     fbk = FDK_Flexible(ig, data.geometry)
    #     fbk.set_input(data)
    #     reconstruction:ImageData | None = fbk.get_output()

    if has_tigre:
        if verbose > 0: print("Backend: Tigre")
        if verbose > 0: print("Use plugin directly: ", use_plugins)

        if use_plugins:
            reconstruction:ImageData | None = FBP_plugin_tigre(ig,data.geometry)(data)
        else:
            reconstruction:ImageData | None = FDK_recons_cil(data, ig).run()
    else:
        if verbose > 0: print("Backend: Astra-Toolbox")
        if verbose > 0: print("Use plugin directly: ", use_plugins)

        if use_plugins:
            reconstruction:ImageData | None = FBP_plugin_astra(ig,data.geometry)(data)
        else:
            raise(ValueError("Not implemented for Astra"))
            # reconstruction:ImageData | None = FDK_recons_cil(data, ig).run() # Not implemented for Astra
            reconstruction:ImageData | None = None

    return reconstruction


def reconstruct(data, source_shape, use_plugin=False, verbose=0):

    if verbose > 0:
        print("Source shape:", source_shape)

    # Use CIL
    if has_cil:

        if verbose > 0: print("Use CIL")

        print("data.geometry", data.geometry)

        if has_tigre:
            data.reorder(order='tigre')
            # data.geometry.set_angles(-data.geometry.angles)
        else:
            data.reorder("astra")
            # data.geometry.set_angles(-data.geometry.angles)

        ig = data.geometry.get_ImageGeometry()

        data_corr = TransmissionAbsorptionConverter(white_level=data.max(), min_intensity=0.000001)(data)

        if type(source_shape) == str:

            if source_shape.upper() == "PARALLELBEAM" or source_shape.upper() == "PARALLEL":
                reconstruction:ImageData | None = reconstructFBPWithCIL(data_corr, ig, use_plugin, verbose)

            elif source_shape.upper() == "POINTSOURCE" or source_shape.upper() == "POINT" or source_shape.upper() == "CONE" or source_shape.upper() == "CONEBEAM":
                reconstruction:ImageData | None = reconstructFDKWithCIL(data_corr, ig, use_plugin, verbose)

            else:
                raise ValueError("Unknown source shape:" + source_shape)

        elif type(source_shape) == type([]):
            if source_shape[0].upper() == "FOCALSPOT":
                reconstruction:ImageData | None = reconstructFDKWithCIL(data_corr, ig, use_plugin, verbose)

            else:
                raise ValueError("Unknown source shape:" + source_shape)

        else:
            raise ValueError("Unknown source shape:" + source_shape)

    else:
        raise ValueError("CIL is not installed")

    return reconstruction