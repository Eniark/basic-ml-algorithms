

def configure_power_scheduler(lr0,every,exponent=1):
    def power_scheduling(iteration):
        return  lr0 / (1 + iteration / every)**exponent
    return power_scheduling


def configure_exponential_decay(lr0, every):
    """Decreases learning rate by a factor of 10 every specified number of epochs"""
    """Parameters"""
    def exponential_decay(iteration):
        return lr0 * 0.1 ** (iteration / every)
    
    return exponential_decay

def configure_piecewise_constant_scheduling(lr, every, multiplier_factor):
    def piecewise_constant_scheduling(epoch):
        if epoch % every==0:
            lr *= multiplier_factor
        return lr
    return piecewise_constant_scheduling

def configure_step_decay(lr0, decay_rate, step_size):
    def step_decay(epoch):
        return lr0 * (1 + epoch // step_size) ** (-decay_rate)
    return step_decay

initial_learning_rate = 0.1
decay_rate = 0.5
step_size = 5





