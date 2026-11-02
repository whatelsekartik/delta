// runtime-native/deltacore_cli.cpp
/**
 * Native C++17 (AVX2 / FMA3) INT8 Inference & Benchmark Engine for DeltaCore-LM.
 * Used in Phase 4 & Phase 5 Audits to:
 * 1. Verify exact numerical logit parity against Python INT8 and WebAssembly INT8.
 * 2. Benchmark native x86_64 AVX2 execution vs V8 WebAssembly execution.
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct QuantizedMatrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<float> scales;
    std::vector<int8_t> data;

    inline void matvec_add(const float* x, float* out) const {
        for (uint32_t r = 0; r < rows; ++r) {
            const int8_t* row_ptr = &data[static_cast<size_t>(r) * cols];
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            uint32_t c = 0;
            for (; c + 3 < cols; c += 4) {
                acc0 += static_cast<float>(row_ptr[c]) * x[c];
                acc1 += static_cast<float>(row_ptr[c + 1]) * x[c + 1];
                acc2 += static_cast<float>(row_ptr[c + 2]) * x[c + 2];
                acc3 += static_cast<float>(row_ptr[c + 3]) * x[c + 3];
            }
            float sum = (acc0 + acc1) + (acc2 + acc3);
            for (; c < cols; ++c) {
                sum += static_cast<float>(row_ptr[c]) * x[c];
            }
            out[r] += sum * scales[r];
        }
    }

    inline void matvec(const float* x, float* out) const {
        std::memset(out, 0, sizeof(float) * rows);
        matvec_add(x, out);
    }
};

struct DepthModulator {
    QuantizedMatrix u;
    QuantizedMatrix v;
    bool has_bias = false;
    std::vector<float> bias;

    inline void apply(
        const float* x,
        const QuantizedMatrix& w_shared,
        float* scratch_rm,
        float* out
    ) const {
        if (has_bias) {
            std::memcpy(out, bias.data(), sizeof(float) * u.rows);
        } else {
            std::memset(out, 0, sizeof(float) * u.rows);
        }
        w_shared.matvec_add(x, out);
        uint32_t rm = u.cols;
        uint32_t d_in = v.rows;
        std::memset(scratch_rm, 0, sizeof(float) * rm);
        for (uint32_t i = 0; i < d_in; ++i) {
            float sx = v.scales[i] * x[i];
            const int8_t* row_ptr = &v.data[static_cast<size_t>(i) * rm];
            for (uint32_t k = 0; k < rm; ++k) {
                scratch_rm[k] += static_cast<float>(row_ptr[k]) * sx;
            }
        }
        u.matvec_add(scratch_rm, out);
    }
};

struct LayerWeights {
    std::vector<float> norm1;
    std::vector<float> norm2;
    std::vector<float> conv_weight;
    DepthModulator mod_u, mod_g, mod_o, mod_gate, mod_up, mod_down;
    QuantizedMatrix w_alpha;
    std::vector<float> b_alpha;
    QuantizedMatrix w_k, w_q, w_v;
    std::vector<float> w_beta;
    float b_beta = 0.0f;
    std::vector<float> w_gamma;
    float b_gamma = 0.0f;
    QuantizedMatrix p_slow;
};

struct LayerState {
    std::vector<float> conv_buf;
    std::vector<float> fast_state;
    std::vector<float> slow_mem;
};

struct SessionState {
    std::vector<LayerState> layers;
    uint64_t rng_state = 1;
    uint32_t tokens_generated = 0;
};

static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float fast_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static inline void rms_norm(const float* x, const float* w, uint32_t n, float eps, float* out) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < n; ++i) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(n) + eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = x[i] * inv_rms * w[i];
}

class DeltaCoreNative {
public:
    uint32_t vocab_size = 0, d_model = 0, d_embed = 0, d_ff = 0;
    uint32_t n_layers = 0, rank_mod = 0, rank_slow = 0, d_value = 0, conv_kernel = 3;
    float rms_eps = 1e-5f;

    QuantizedMatrix tok_embed, p_in, p_out;
    QuantizedMatrix core_wu, core_wg, core_wo, core_wgate, core_wup, core_wdown;
    std::vector<LayerWeights> layers;
    std::vector<float> final_norm;

    bool load_from_bytes(const std::vector<uint8_t>& buf) {
        if (buf.size() < 48 || std::memcmp(buf.data(), "DCOR", 4) != 0) return false;
        size_t pos = 4;
        auto read_u32 = [&]() -> uint32_t {
            uint32_t v;
            std::memcpy(&v, &buf[pos], 4);
            pos += 4;
            return v;
        };
        auto read_f32 = [&]() -> float {
            float v;
            std::memcpy(&v, &buf[pos], 4);
            pos += 4;
            return v;
        };
        auto read_f32_vec = [&](uint32_t n) -> std::vector<float> {
            std::vector<float> out(n);
            std::memcpy(out.data(), &buf[pos], sizeof(float) * n);
            pos += sizeof(float) * n;
            return out;
        };
        auto read_i8_mat = [&](uint32_t r, uint32_t c) -> QuantizedMatrix {
            QuantizedMatrix m;
            m.rows = r;
            m.cols = c;
            m.scales = read_f32_vec(r);
            m.data.resize(static_cast<size_t>(r) * c);
            std::memcpy(m.data.data(), &buf[pos], static_cast<size_t>(r) * c);
            pos += static_cast<size_t>(r) * c;
            return m;
        };
        auto read_mod = [&](uint32_t d_in, uint32_t d_out, uint32_t rm, bool has_b) -> DepthModulator {
            DepthModulator dm;
            dm.u = read_i8_mat(d_out, rm);
            dm.v = read_i8_mat(d_in, rm);
            dm.has_bias = has_b;
            if (has_b) dm.bias = read_f32_vec(d_out);
            return dm;
        };

        uint32_t version = read_u32();
        if (version != 1) return false;
        vocab_size = read_u32();
        d_model = read_u32();
        d_embed = read_u32();
        d_ff = read_u32();
        n_layers = read_u32();
        rank_mod = read_u32();
        rank_slow = read_u32();
        d_value = read_u32();
        rms_eps = read_f32();
        conv_kernel = read_u32();

        tok_embed = read_i8_mat(vocab_size, d_embed);
        p_in = read_i8_mat(d_model, d_embed);
        p_out = read_i8_mat(d_embed, d_model);

        core_wu = read_i8_mat(d_model, d_model);
        core_wg = read_i8_mat(d_model, d_model);
        core_wo = read_i8_mat(d_model, d_model);
        core_wgate = read_i8_mat(d_ff, d_model);
        core_wup = read_i8_mat(d_ff, d_model);
        core_wdown = read_i8_mat(d_model, d_ff);

        layers.resize(n_layers);
        for (uint32_t l = 0; l < n_layers; ++l) {
            auto& lw = layers[l];
            lw.norm1 = read_f32_vec(d_model);
            lw.norm2 = read_f32_vec(d_model);
            lw.conv_weight = read_f32_vec(conv_kernel * d_model);
            lw.mod_u = read_mod(d_model, d_model, rank_mod, true);
            lw.mod_g = read_mod(d_model, d_model, rank_mod, true);
            lw.mod_o = read_mod(d_model, d_model, rank_mod, false);
            lw.mod_gate = read_mod(d_model, d_ff, rank_mod, false);
            lw.mod_up = read_mod(d_model, d_ff, rank_mod, false);
            lw.mod_down = read_mod(d_ff, d_model, rank_mod, false);
            lw.w_alpha = read_i8_mat(d_model, d_model);
            lw.b_alpha = read_f32_vec(d_model);
            lw.w_k = read_i8_mat(rank_slow, d_model);
            lw.w_q = read_i8_mat(rank_slow, d_model);
            lw.w_v = read_i8_mat(d_value, d_model);
            lw.w_beta = read_f32_vec(d_model);
            lw.b_beta = read_f32();
            lw.w_gamma = read_f32_vec(d_model);
            lw.b_gamma = read_f32();
            lw.p_slow = read_i8_mat(d_model, d_value);
        }
        final_norm = read_f32_vec(d_model);
        return pos == buf.size();
    }

    SessionState new_session(uint64_t seed) const {
        SessionState s;
        s.rng_state = seed > 0 ? seed : 1;
        s.layers.resize(n_layers);
        for (uint32_t l = 0; l < n_layers; ++l) {
            s.layers[l].conv_buf.assign((conv_kernel - 1) * d_model, 0.0f);
            s.layers[l].fast_state.assign(d_model, 0.0f);
            s.layers[l].slow_mem.assign(rank_slow * d_value, 0.0f);
        }
        return s;
    }

    void forward_step(uint32_t token_id, SessionState& session, float* logits_out) const {
        std::vector<float> e_vec(d_embed), h(d_model), z(d_model), z_shift(d_model);
        std::vector<float> scratch_rm(rank_mod), u_vec(d_model), g_vec(d_model), alpha_vec(d_model);
        std::vector<float> k_vec(rank_slow), q_vec(rank_slow), v_vec(d_value), s_vec(d_value);
        std::vector<float> combined(d_model), o_proj(d_model), ff_gate(d_ff), ff_up(d_ff);

        float emb_scale = tok_embed.scales[token_id];
        const int8_t* emb_row = &tok_embed.data[static_cast<size_t>(token_id) * d_embed];
        for (uint32_t i = 0; i < d_embed; ++i) {
            e_vec[i] = static_cast<float>(emb_row[i]) * emb_scale;
        }
        p_in.matvec(e_vec.data(), h.data());

        for (uint32_t l = 0; l < n_layers; ++l) {
            const auto& lw = layers[l];
            auto& st = session.layers[l];
            rms_norm(h.data(), lw.norm1.data(), d_model, rms_eps, z.data());

            for (uint32_t i = 0; i < d_model; ++i) {
                z_shift[i] = z[i] * lw.conv_weight[i];
            }
            for (uint32_t tau = 1; tau < conv_kernel; ++tau) {
                uint32_t buf_slot = (conv_kernel - 1 - tau) * d_model;
                uint32_t w_slot = tau * d_model;
                for (uint32_t i = 0; i < d_model; ++i) {
                    z_shift[i] += st.conv_buf[buf_slot + i] * lw.conv_weight[w_slot + i];
                }
            }
            if (conv_kernel > 2) {
                std::memmove(
                    st.conv_buf.data(),
                    st.conv_buf.data() + d_model,
                    sizeof(float) * (conv_kernel - 2) * d_model
                );
            }
            std::memcpy(
                st.conv_buf.data() + (conv_kernel - 2) * d_model,
                z.data(),
                sizeof(float) * d_model
            );

            std::memcpy(alpha_vec.data(), lw.b_alpha.data(), sizeof(float) * d_model);
            lw.w_alpha.matvec_add(z_shift.data(), alpha_vec.data());
            lw.mod_u.apply(z_shift.data(), core_wu, scratch_rm.data(), u_vec.data());
            for (uint32_t i = 0; i < d_model; ++i) {
                float a = fast_sigmoid(alpha_vec[i]);
                float u = std::tanh(u_vec[i]);
                st.fast_state[i] = a * st.fast_state[i] + (1.0f - a) * u;
            }

            lw.w_k.matvec(z_shift.data(), k_vec.data());
            float kn_sq = 0.0f;
            for (uint32_t i = 0; i < rank_slow; ++i) kn_sq += k_vec[i] * k_vec[i];
            float inv_kn = 1.0f / (std::sqrt(kn_sq) + 1e-6f);
            for (uint32_t i = 0; i < rank_slow; ++i) k_vec[i] *= inv_kn;

            lw.w_q.matvec(z_shift.data(), q_vec.data());
            float inv_sqrt_rs = 1.0f / std::sqrt(static_cast<float>(rank_slow));
            for (uint32_t i = 0; i < rank_slow; ++i) q_vec[i] *= inv_sqrt_rs;

            lw.w_v.matvec(z_shift.data(), v_vec.data());

            float beta_dot = lw.b_beta, gamma_dot = lw.b_gamma;
            for (uint32_t i = 0; i < d_model; ++i) {
                beta_dot += lw.w_beta[i] * z_shift[i];
                gamma_dot += lw.w_gamma[i] * z_shift[i];
            }
            float beta = fast_sigmoid(beta_dot);
            float gamma = fast_sigmoid(gamma_dot);

            for (uint32_t j = 0; j < d_value; ++j) {
                float v_hat_j = 0.0f;
                for (uint32_t r = 0; r < rank_slow; ++r) {
                    uint32_t idx = r * d_value + j;
                    float m_dec = gamma * st.slow_mem[idx];
                    st.slow_mem[idx] = m_dec;
                    v_hat_j += m_dec * k_vec[r];
                }
                float delta_j = beta * (v_vec[j] - v_hat_j);
                float readout_j = 0.0f;
                for (uint32_t r = 0; r < rank_slow; ++r) {
                    uint32_t idx = r * d_value + j;
                    float m_new = st.slow_mem[idx] + k_vec[r] * delta_j;
                    st.slow_mem[idx] = m_new;
                    readout_j += m_new * q_vec[r];
                }
                s_vec[j] = readout_j;
            }

            lw.mod_g.apply(z_shift.data(), core_wg, scratch_rm.data(), g_vec.data());
            std::memcpy(combined.data(), st.fast_state.data(), sizeof(float) * d_model);
            lw.p_slow.matvec_add(s_vec.data(), combined.data());
            for (uint32_t i = 0; i < d_model; ++i) {
                combined[i] *= fast_silu(g_vec[i]);
            }
            lw.mod_o.apply(combined.data(), core_wo, scratch_rm.data(), o_proj.data());
            for (uint32_t i = 0; i < d_model; ++i) h[i] += o_proj[i];

            rms_norm(h.data(), lw.norm2.data(), d_model, rms_eps, z.data());
            lw.mod_gate.apply(z.data(), core_wgate, scratch_rm.data(), ff_gate.data());
            lw.mod_up.apply(z.data(), core_wup, scratch_rm.data(), ff_up.data());
            for (uint32_t i = 0; i < d_ff; ++i) {
                ff_gate[i] = fast_silu(ff_gate[i]) * ff_up[i];
            }
            lw.mod_down.apply(ff_gate.data(), core_wdown, scratch_rm.data(), o_proj.data());
            for (uint32_t i = 0; i < d_model; ++i) h[i] += o_proj[i];
        }

        rms_norm(h.data(), final_norm.data(), d_model, rms_eps, z.data());
        p_out.matvec(z.data(), e_vec.data());
        tok_embed.matvec(e_vec.data(), logits_out);
        session.tokens_generated += 1;
    }
};

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "artifacts/model.dcore";
    std::string mode = (argc > 2) ? argv[2] : "bench";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << path << "\n";
        return 1;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    DeltaCoreNative model;
    if (!model.load_from_bytes(buf)) {
        std::cerr << "Invalid .dcore binary\n";
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (mode == "logits") {
        // Output JSON array of logits after feeding comma-separated token_ids in argv[3]
        auto session = model.new_session(42);
        std::vector<float> logits(model.vocab_size, 0.0f);
        std::string seq_str = (argc > 3) ? argv[3] : "10,20,30";
        size_t pos = 0;
        while (pos < seq_str.size()) {
            size_t comma = seq_str.find(',', pos);
            std::string sub = (comma == std::string::npos) ? seq_str.substr(pos) : seq_str.substr(pos, comma - pos);
            uint32_t tid = static_cast<uint32_t>(std::stoul(sub));
            model.forward_step(tid, session, logits.data());
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        std::printf("[");
        for (uint32_t i = 0; i < model.vocab_size; ++i) {
            std::printf("%s%.7f", (i == 0 ? "" : ","), logits[i]);
        }
        std::printf("]\n");
        return 0;
    }

    // Benchmark mode across context lengths
    auto session = model.new_session(42);
    std::vector<float> logits(model.vocab_size, 0.0f);
    // Warmup 64 steps
    for (int i = 0; i < 64; ++i) {
        model.forward_step(static_cast<uint32_t>(10 + (i % 50)), session, logits.data());
    }
    int n_steps = 512;
    auto tb0 = std::chrono::high_resolution_clock::now();
    uint32_t tok = 12;
    for (int i = 0; i < n_steps; ++i) {
        model.forward_step(tok, session, logits.data());
        uint32_t best = 0;
        float best_v = -1e30f;
        for (uint32_t v = 0; v < model.vocab_size; ++v) {
            if (logits[v] > best_v) {
                best_v = logits[v];
                best = v;
            }
        }
        tok = best;
    }
    auto tb1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
    double ms_per_tok = total_ms / n_steps;
    double tok_per_sec = 1000.0 / ms_per_tok;

    std::printf(
        "{\"engine\":\"native_cpp_avx2\",\"artifact_bytes\":%zu,\"cold_start_ms\":%.3f,\"ms_per_token\":%.4f,\"tokens_per_sec\":%.1f}\n",
        buf.size(), load_ms, ms_per_tok, tok_per_sec
    );
    return 0;
}
