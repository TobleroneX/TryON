import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('App Integration Tests', () => {
    beforeEach(() => {
        mockFetch.mockClear();
    });

    test('successful image upload and processing flow', async () => {
        // Mock successful responses
        mockFetch
            // First upload (person)
            .mockImplementationOnce(() => 
                Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ filename: 'person.jpg' })
                })
            )
            // Second upload (garment)
            .mockImplementationOnce(() => 
                Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ filename: 'garment.jpg' })
                })
            )
            // Processing
            .mockImplementationOnce(() => 
                Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ 
                        result: 'processed_image_data'
                    })
                })
            );

        render(<App />);

        // Upload person image
        const personInput = screen.getByLabelText(/Upload Person Image/i);
        const file = new File(['dummy content'], 'person.jpg', { type: 'image/jpeg' });
        await userEvent.upload(personInput, file);

        // Upload garment image
        const garmentInput = screen.getByLabelText(/Upload Garment Image/i);
        await userEvent.upload(garmentInput, file);

        // Click process button
        const processButton = screen.getByText(/Try On Garment/i);
        await userEvent.click(processButton);

        // Verify loading states
        expect(screen.getByText(/Processing/i)).toBeInTheDocument();

        // Wait for result
        await waitFor(() => {
            expect(screen.getByAltText(/Try-on result/i)).toBeInTheDocument();
        });
    });

    test('handles upload errors', async () => {
        // Mock failed upload
        mockFetch.mockImplementationOnce(() => 
            Promise.resolve({
                ok: false,
                json: () => Promise.resolve({ 
                    error: 'File too large' 
                })
            })
        );

        render(<App />);

        const input = screen.getByLabelText(/Upload Person Image/i);
        const file = new File(['dummy content'], 'large.jpg', { type: 'image/jpeg' });
        await userEvent.upload(input, file);

        await waitFor(() => {
            expect(screen.getByText(/File too large/i)).toBeInTheDocument();
        });
    });

    test('handles API timeout', async () => {
        mockFetch
            // Successful uploads
            .mockImplementationOnce(() => 
                Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ filename: 'person.jpg' })
                })
            )
            .mockImplementationOnce(() => 
                Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ filename: 'garment.jpg' })
                })
            )
            // Processing timeout
            .mockImplementationOnce(() => 
                new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('timeout')), 100)
                )
            );

        render(<App />);

        // Upload both images
        const file = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
        await userEvent.upload(screen.getByLabelText(/Upload Person Image/i), file);
        await userEvent.upload(screen.getByLabelText(/Upload Garment Image/i), file);

        // Try processing
        await userEvent.click(screen.getByText(/Try On Garment/i));

        await waitFor(() => {
            expect(screen.getByText(/timeout/i)).toBeInTheDocument();
        });
    });

    test('handles rate limit errors', async () => {
        mockFetch.mockImplementationOnce(() => 
            Promise.resolve({
                ok: false,
                status: 429,
                json: () => Promise.resolve({ 
                    error: 'Rate limit exceeded' 
                })
            })
        );

        render(<App />);

        const input = screen.getByLabelText(/Upload Person Image/i);
        const file = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
        await userEvent.upload(input, file);

        await waitFor(() => {
            expect(screen.getByText(/Rate limit exceeded/i)).toBeInTheDocument();
        });
    });

    test('handles network errors', async () => {
        mockFetch.mockImplementationOnce(() => 
            Promise.reject(new Error('Network error'))
        );

        render(<App />);

        const input = screen.getByLabelText(/Upload Person Image/i);
        const file = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
        await userEvent.upload(input, file);

        await waitFor(() => {
            expect(screen.getByText(/Network error/i)).toBeInTheDocument();
        });
    });

    test('disables process button during upload', async () => {
        render(<App />);

        const processButton = screen.getByText(/Try On Garment/i);
        expect(processButton).toBeDisabled();

        // Upload one image
        const file = new File(['dummy content'], 'test.jpg', { type: 'image/jpeg' });
        await userEvent.upload(screen.getByLabelText(/Upload Person Image/i), file);

        // Button should still be disabled
        expect(processButton).toBeDisabled();

        // Upload second image
        await userEvent.upload(screen.getByLabelText(/Upload Garment Image/i), file);

        // Button should be enabled
        expect(processButton).not.toBeDisabled();
    });
}); 