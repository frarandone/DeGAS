def get_program(name):
    if name == "bernoulli":
        x = """
        theta = uniform([0,1], 2);
        
        if theta < _p {
            y = 1.;
        } else { 
            y = 0.; 
        } end if;
        """

    elif name == "burglary":
        x = """
        theta = uniform([0,1], 2);
        if theta < _pe {
            earthquake = 1.;
        } else { 
            earthquake = 0.; 
        } end if;

        if theta < _pb {
            burglary = 1.;
        } else {
            burglary = 0.; 
        } end if;

        if earthquake == 0 {
            if burglary == 0 {
                alarm = 0;
            } else {
                alarm = 1;
            } end if;
        } else {
            alarm = 1;
        } end if;

        if earthquake == 1 {
            phoneWorking = gm([0.7, 0.3], [1., 0.], [0., 0.]);
        } else {
            phoneWorking = gm([0.99, 0.01], [1., 0.], [0., 0.]);
        } end if;

        if alarm == 1 {
            if earthquake == 1 {
                maryWakes = gm([0.8, 0.2], [1., 0.], [0., 0.]);
            } else {
                maryWakes = gm([0.6, 0.4], [1., 0.], [0., 0.]);
            } end if;
        } else {
            maryWakes = gm([0.2, 0.8], [1., 0.], [0., 0.]);
        } end if;

        if maryWakes == 1 {
            if phoneWorking == 1 {
                called = 1;
            } else {
                called = 0;
            } end if;
        } else {
            called = 0;
        } end if;"""

    elif name == "clickgraph":
        x = """
        simAll = uniform([0,1],2);

        if simAll < _p {
            sim = 1;
        } else {
            sim = 0;
        } end if;

        beta1 = uniform([0,1],2);
        if sim == 1 {
            beta2 = beta1;
        } else {
            beta2 = uniform([0,1],2);
        } end if;

        if  uniform([0,1],2) -  beta1 < 0 {
            click0 = 1;
        } else {
            click0 = 0;
        } end if;


        if  uniform([0,1],2) -  beta2 < 0 {
            click1 = 1;
        } else {
            click1 = 0;
        } end if;

        """
    else:
        raise ValueError("Program not recognized")

    return x