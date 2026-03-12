use ff_ext::ExtensionField;
use p3::field::{Field, TwoAdicField};
use p3_field::{PrimeCharacteristicRing, coset::TwoAdicMultiplicativeCoset};

#[derive(Debug, Clone)]
pub struct Domain<E>
where
    E: ExtensionField,
{
    pub base_domain: Option<TwoAdicMultiplicativeCoset<E::BaseField>>, // base-field domain for initial FFT
    pub backing_domain: TwoAdicMultiplicativeCoset<E>,
}

impl<E> Domain<E>
where
    E: ExtensionField,
{
    pub fn new(degree: usize, log_rho_inv: usize) -> Option<Self> {
        let size = degree * (1 << log_rho_inv);
        let log_size = p3::util::log2_strict_usize(size);
        let base_domain =
            TwoAdicMultiplicativeCoset::new(E::BaseField::from_u64(1), log_size)?;
        let backing_domain = TwoAdicMultiplicativeCoset::new(E::ONE, log_size)
            .expect("extension field must support the same two-adicity");

        Some(Self { base_domain: Some(base_domain), backing_domain })
    }

    // returns the size of the domain after folding folding_factor many times.
    //
    // This asserts that the domain size is divisible by 1 << folding_factor
    pub fn folded_size(&self, folding_factor: usize) -> usize {
        let log_size = self.backing_domain.log_size();
        assert!(log_size >= folding_factor);
        1 << (log_size - folding_factor)
    }

    pub fn size(&self) -> usize {
        self.backing_domain.size()
    }

    pub fn scale(&self, power: usize) -> Self {
        debug_assert!(power.is_power_of_two(), "scale expects a power-of-two factor");
        let log_power = p3::util::log2_strict_usize(power);
        let backing_domain = self
            .backing_domain
            .shrink_coset(log_power)
            .expect("folding factor exceeds domain size");
        Self {
            backing_domain,
            base_domain: None,
        }
    }

    pub fn backing_domain_group_gen(&self) -> E {
        E::two_adic_generator(self.backing_domain.log_size())
    }

    pub fn base_domain_group_gen(&self) -> E::BaseField {
        E::BaseField::two_adic_generator(self.backing_domain.log_size())
    }

    pub fn base_domain_group_gen_inv(&self) -> E::BaseField {
        self.base_domain_group_gen().inverse()
    }

    pub fn backing_domain_element(&self, index: usize) -> E {
        E::two_adic_generator(self.backing_domain.log_size()).exp_u64(index as u64)
    }

    pub fn backing_domain_element_pow_of_2(&self, exp: usize) -> E {
        let log_size = self.backing_domain.log_size();
        assert!(exp <= log_size);
        E::two_adic_generator(log_size - exp)
    }
}
